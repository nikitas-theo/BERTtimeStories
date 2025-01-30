import math
import os
import time
from socket import gethostname

import torch
import torch.utils
import yaml
from tokenizers import Tokenizer
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import PreTrainedTokenizerFast

from baby_lm.train_config import WANDB_KEY, get_config
from baby_lm.train_utils import (
    calculate_loss,
    estimate_loss,
    get_grad_norm,
    get_model_summary,
    get_optimizer_and_scheduler,
    load_data_long_after,
    load_dataset_dataloader,
    monitor_CPU_memory,
    monitor_CUDA_memory,
    prepare_model,
    seed_everything,
    setup_optimization_params,
)


def do_train():
    config = get_config()
    if config.DEBUG:
        print("DEBUG MODE ON")

    # print config for logging
    print("=" * 100)
    print(yaml.dump(config.__dict__), flush=True)
    print("=" * 100)

    # if load_checkpoint is set resume training
    if config.load_checkpoint is not None:
        print(f"loading config from {config.load_checkpoint}", flush=True)
        checkpoint_path = config.load_checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        ############################
        checkpoint_config = checkpoint["config"]

        # We won't load old configuration except for these
        config.wandb_id = checkpoint_config.wandb_id
        config.run_name = checkpoint_config.run_name
        config.save_dir = checkpoint_config.save_dir
        #############################

        # check difference between current config and checkpoint config for sanity
        dict_checkpoint_config = checkpoint_config.__dict__
        dict_config = config.__dict__
        only_in_dict_checkpoint_config = {
            k: dict_checkpoint_config[k]
            for k in dict_checkpoint_config
            if k not in dict_config or dict_checkpoint_config[k] != dict_config[k]
        }
        only_in_dict_config = {
            k: dict_config[k]
            for k in dict_config
            if k not in dict_checkpoint_config
            or dict_checkpoint_config[k] != dict_config[k]
        }

        # print differences
        print(
            "Items only in dict_checkpoint_config:",
            only_in_dict_checkpoint_config,
            flush=True,
        )
        print("Items only in dict_config:", only_in_dict_config, flush=True)

    else:
        checkpoint = None

    gradient_accumulation_steps = config.gradient_accumulation_steps

    # handle DDP training
    # MASTER_ADDR, MASTER_PORT, WORLD_SIZE, LOCAL_RANK need to be set
    # SLURM_GPUS_ON_NODE is optional
    ############################################################
    ddp = False
    try:
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        print(int(os.environ["WORLD_SIZE"]))
    except KeyError:
        ddp_world_size = 1

    if ddp_world_size > 1:
        ddp = True

    print("Distributed Data Parallel:", ddp, flush=True)
    if ddp:
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        print("initing", flush=True)
        print("World size ", ddp_world_size, flush=True)

        ddp_rank = int(os.environ["RANK"])
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]

        print("My rank is ", ddp_rank, flush=True)

        init_process_group(
            backend="nccl",
            rank=ddp_rank,
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=ddp_world_size,
        )

        print("did init", flush=True)
        if os.environ["SLURM_GPUS_ON_NODE"] is not None:
            gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
            assert gpus_per_node == torch.cuda.device_count()
        else:
            gpus_per_node = torch.cuda.device_count()

        ddp_local_rank = int(os.environ["LOCAL_RANK"])

        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
        n_gpus = ddp_world_size
        offset = ddp_rank

        print(
            f"Hello from rank {ddp_rank} of {ddp_world_size} on {gethostname()} where \
                there are {gpus_per_node} allocated GPUs per node.",
            flush=True,
        )

        if ddp_rank == 0:
            print(
                f"Group initialized? {torch.distributed.is_initialized()}", flush=True
            )

        config.device = torch.device("cuda", ddp_local_rank)
        print(f"Device set to {config.device}", flush=True)
    else:
        # if not DDP training
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        n_gpus = 1
        offset = 0

    ############################################################
    # set seed
    seed_everything(config.seed + seed_offset)

    # prepare model
    ############################################################

    model, model_config = prepare_model(config)

    # load model from checkpoint
    # fix DDP model conversion error if needed
    if checkpoint is not None:
        try:
            model.load_state_dict(checkpoint["model"])
        except:
            new_state_dict = {}
            for key in checkpoint["model"]:
                if key.startswith("module."):
                    new_key = key[len("module.") :]  # remove 'module.' from the key
                    new_state_dict[new_key] = checkpoint["model"][key]
                else:
                    new_state_dict[key] = checkpoint["model"][key]

            model_state_dict = new_state_dict
            model.load_state_dict(model_state_dict)

    # should be done before optimizer
    model.train()
    model.to(config.device)

    # load DDP model
    if ddp:
        model = DDP(
            model,
            device_ids=[ddp_local_rank],
            # set to the total memory of GPU, reduce overhead of communication
            # but might require more memory
            # bucket_cap_mb=torch.cuda.get_device_properties(device).total_memory,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            # static_graph=True,
        )

    # load dataset, tokenizer
    ############################################################

    tokenizer = Tokenizer.from_file(config.tokenizer_path)

    dataset, dataloader = load_dataset_dataloader(
        config,
        config.cached_data_file,
        tokenizer,
        config.seq_length,
        config.batch_size,
        num_workers=config.num_workers,
        n_gpus=n_gpus,
        offset=offset,
        random_sampling=config.random_sampling,
    )

    # if master, load validation data
    if master_process:
        dataset_val, dataloader_val = load_dataset_dataloader(
            config,
            config.cached_valid_data_file,
            tokenizer,
            config.seq_length,
            config.batch_size,
            num_workers=config.num_workers,
            n_gpus=n_gpus,
            offset=offset,
            random_sampling=config.random_sampling,
        )

    # training setup
    ############################################################

    # epochs parameter overrides max_steps (if it's not None)
    if config.epochs is not None:
        config.max_steps = math.ceil(
            config.epochs * len(dataloader) / gradient_accumulation_steps
        )

    # training statistics
    if master_process:
        # print steps for 1 epoch
        steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Training for {config.max_steps / steps_per_epoch:.2f} epochs.")
        # get processed tokens per iteration
        tokens_per_iter = (
            gradient_accumulation_steps
            * config.batch_size
            * config.seq_length
            * ddp_world_size
        )
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"training for {config.max_steps} steps")

    # optimization setup
    optimizer, scheduler = get_optimizer_and_scheduler(config, model)
    scaler, ctx = setup_optimization_params(config)

    # initial setup and loading
    steps = 0
    best_val_loss = 1e9

    # wandb
    if config.wandb_log and master_process:
        import wandb

    # resume training: load optimizer, scaler and lr scheduler
    if checkpoint is not None:
        if master_process:
            print(f"loading checkpoint from {checkpoint_path}.")
            print(f"wandb is enabled: {config.wandb_log}.")

        # get steps, model, optimizer, scheduler, scaler
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        best_val_loss = checkpoint["best_val_loss"]
        steps = checkpoint["steps"] + 1
    else:
        # generate a new id if we don't load from checkpoint
        # also save to config
        if master_process and config.wandb_log:
            config.wandb_id = wandb.util.generate_id()

    # wandb setup
    if master_process:
        if config.wandb_log:
            if WANDB_KEY is not None:
                wandb.login(key=WANDB_KEY)

            mode = "offline" if config.wandb_offline else "online"

            wandb.init(
                project=config.wandb_project,
                name=config.run_name,
                config=config,
                id=config.wandb_id,
                mode=mode,
                resume="allow",
                dir=config.output_dir,
            )

    if master_process:
        if config.wandb_log:
            save_dir = wandb.run.dir.replace("files", "checkpoint")
            config.save_dir = save_dir
            wandb.config.update({"save_dir": save_dir}, allow_val_change=True)
        else:
            save_dir = config.output_dir + "/" + config.run_name + "/checkpoint/"
            config.save_dir = save_dir

    # save the tokenizer
    # reload tokenizer to avoid error
    if master_process:
        tokenizer_save = PreTrainedTokenizerFast(tokenizer_file=config.tokenizer_path)
        tokenizer_save.save_pretrained(config.save_dir)

    # get model summary
    if master_process and config.wandb_log:
        pytorch_total_params = get_model_summary(model, config)
        wandb.config.update({"n_params": pytorch_total_params})

    # torch compile
    if config.compile and master_process:
        # Python 3.12 does not support compile
        try:
            print("compiling the model... (takes a ~minute)")
            model = torch.compile(model)  # requires PyTorch 2.0
        except RuntimeError as e:
            print(e)
            print("not compiling...")
        except AttributeError as e:
            print(e)
            print("cant compile with torch <= 2.0..", flush=True)

    # need to add >= here so we make sure it works in checkpoint continuation
    if config.long_after is not None:
        if steps >= int(config.max_steps * config.long_after):
            (
                dataset,
                dataloader,
                dataset_val,
                dataloader_val,
                gradient_accumulation_steps,
            ) = load_data_long_after(
                config,
                steps,
                tokenizer,
                master_process,
                n_gpus=n_gpus,
                offset=offset,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )

    print("", flush=True)
    t0 = time.time()
    grad_norms_list = []
    dt_list = []

    ################################################################
    # main training loop
    ################################################################
    while True:
        for i, batch in enumerate(dataloader):
            if config.DEBUG:
                print(f"Batch {batch[0].shape}", flush=True)
                print(f"grad accum {gradient_accumulation_steps}", flush=True)
                print(f"steps {steps}", flush=True)

            # condition, check if we need to do a gradient_step
            do_gradient_step = (i + 1) % gradient_accumulation_steps == 0
            do_gradient_step = do_gradient_step or i == len(dataloader) - 1
            did_gradient_step = i % gradient_accumulation_steps == 0

            # because we step at the start
            # we need to step right after a grad step happened
            if did_gradient_step:
                # set learning rate
                # using custom scheduler and step()-ing at the start of the loop
                lr = scheduler.step()
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            # need this for the gradient accumulation while multi-gpu
            # this code can be refactored for more clarity
            if (not ddp) or do_gradient_step:
                # anyway we pass through model
                # forward pass with autocast for lower precision
                with ctx:
                    loss, accuracy = calculate_loss(batch, model, config)
                    loss = loss / gradient_accumulation_steps
                    # scale loss for fp16
                    scaler.scale(loss).backward()
                    if config.DEBUG:
                        print("Scaler ", scaler.get_scale(), flush=True)

            else:
                with ctx:
                    with model.no_sync():
                        loss, accuracy = calculate_loss(batch, model, config)
                        loss = loss / gradient_accumulation_steps
                        # scale loss for fp16
                        scaler.scale(loss).backward()

            # it's time to do a gradient step
            if do_gradient_step:
                if config.DEBUG:
                    print("Gradient Step", flush=True)

                # evaluate and checkpoint
                do_evaluate = (steps + 1) % config.eval_every == 0
                if do_evaluate and master_process:
                    print("Evaluating..")
                    losses = estimate_loss(
                        model, dataloader, dataloader_val, ctx, config
                    )

                    # logging on wandb
                    if config.wandb_log:
                        mean_grad_norm = torch.tensor(grad_norms_list).mean()
                        grad_norms_list = []

                        wandb.log(
                            {
                                "train/loss": losses["train"],
                                "train/accuracy": losses["train_accuracy"],
                                "val/loss": losses["val"],
                                "val/accuracy": losses["val_accuracy"],
                                "stats/grad_norm": mean_grad_norm,
                                "stats/seq_length": dataset.seq_length,
                            },
                            step=steps,
                        )

                    else:
                        print(
                            (
                                f"iter {steps}: train loss {losses['train']}"
                                f"val loss {losses['val']}"
                            )
                        )

                    # checkpoint best model, or at every evaluation
                    print(f"Current loss is: {losses['val'].item():.5f}")
                    print(f"Best loss is: {best_val_loss:.5f}")

                    if losses["val"] < best_val_loss or config.always_save_checkpoint:
                        best_val_loss = losses["val"].item()

                        print("Saving checkpoint..")
                        # don't save on start
                        if steps > 0:
                            # saving model checkpoint
                            os.makedirs(config.save_dir, exist_ok=True)
                            checkpoint_path = f"{config.save_dir}/checkpoint.pth"
                            torch.save(
                                {
                                    "model": model.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "scaler": scaler.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                    "steps": steps,
                                    "config": config,
                                    "model_config": model_config,
                                    "best_val_loss": best_val_loss,
                                },
                                checkpoint_path,
                            )

                            # for GPT-Neo, save model also using HuggingFace API
                            if config.model_type == "gpt":
                                if ddp:
                                    model.module.save_pretrained(config.save_dir)
                                else:
                                    model.save_pretrained(config.save_dir)

                            print(f"Saving checkpoint to {checkpoint_path}")

                # clip gradients
                if config.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    )
                else:
                    grad_norm = get_grad_norm(model)

                # avoid memory leak
                grad_norm = grad_norm.cpu().detach().item()
                grad_norms_list.append(grad_norm)

                # training step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # print useful statistics
                do_print = (steps + 1) % config.print_every == 0
                if master_process and do_print:
                    t1 = time.time()
                    dt = t1 - t0
                    t0 = t1
                    if config.DEBUG:
                        dt_list.append(dt)

                    if config.DEBUG:
                        monitor_CPU_memory()
                        monitor_CUDA_memory()

                    epoch = steps // steps_per_epoch + 1
                    lossf = loss.item() * gradient_accumulation_steps
                    print(
                        (
                            f"steps {steps} - epoch {epoch} - loss: {lossf:.5f} - acc: {accuracy:.5f}"
                            f" - lr: {optimizer.param_groups[0]['lr']:.8f} - time {dt * 1000:.2f}ms"
                            f" - grad norm: {grad_norm:.2f}"
                        ),
                        flush=True,
                    )
                    if config.wandb_log:
                        wandb.log(
                            {
                                "iter/loss": lossf,
                                "iter/accuracy": accuracy,
                                "iter/lr": optimizer.param_groups[0]["lr"],
                                "iter/time": dt,
                                "iter/grad_norm": grad_norm,
                                "iter/epoch": epoch,
                            },
                            step=steps,
                        )

                # update steps
                steps += 1

                # increase sequence length, if applicable
                if config.long_after is not None:
                    if steps == int(config.max_steps * config.long_after):
                        dataset, dataloader, dataset_val, dataloader_val = (
                            load_data_long_after(
                                config,
                                steps,
                                tokenizer,
                                master_process,
                                n_gpus=n_gpus,
                                offset=offset,
                                gradient_accumulation_steps=gradient_accumulation_steps,
                            )
                        )
                        # init loop dataloader again
                        break

                # break when needed
                if steps >= config.max_steps:
                    break_all = True
                    break
                else:
                    break_all = False

        if break_all:
            break

    # cleanup
    if config.wandb_log:
        if master_process:
            wandb.run.finish()
    if ddp:
        destroy_process_group()
    if config.DEBUG and master_process:
        print(dt_list)
        print("Average dt:", sum(dt_list) / len(dt_list))


if __name__ == "__main__":
    do_train()
    print("#" * 30)
    print("DONE.")
