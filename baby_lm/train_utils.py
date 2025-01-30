import math
import os
import random
from contextlib import nullcontext

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchinfo import summary
from transformers import AutoConfig, AutoModelForCausalLM

from baby_lm.encoder.dataset import BERTDataset
from baby_lm.encoder.ltg_config import LTGBERTConfig
from baby_lm.encoder.ltg_model import LTGBERT
from baby_lm.generator.dataset import GPTDataset


class ConstScheduler:
    def __init__(self, lr):
        self.lr = lr

    def step(self):
        return self.lr

    def load_state_dict(self, json_object):
        self.__init__(**json_object)

    def state_dict(self):
        output = self.__dict__
        return output


class LrScheduler:
    def __init__(self, warmup_steps, max_lr, min_lr, max_steps, _step=0):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_steps = max_steps
        self._step = _step

    def _get_lr(self, it):
        if it < self.warmup_steps:
            return self.max_lr * (it) / self.warmup_steps

        if it > self.max_steps:
            return self.min_lr

        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def step(self):
        self._step += 1
        lr = self._get_lr(self._step)
        return lr

    def load_state_dict(self, json_object):
        self.__init__(**json_object)

    def state_dict(self):
        output = self.__dict__
        return output


@torch.no_grad()
def estimate_loss(model, train_dataloader, val_dataloader, ctx, config):
    """
    estimate training and validation loss
    """

    splits = ["train", "val"]
    dataloaders = [train_dataloader, val_dataloader]
    out = {}
    model.eval()

    for split, dataloader in zip(splits, dataloaders):
        losses = torch.zeros(config.eval_iters)
        accuracies = torch.zeros(config.eval_iters)
        print(f"Evaluating {split}.")

        for k, batch in enumerate(dataloader):
            if k == config.eval_iters:
                break

            with ctx:
                loss, accuracy = calculate_loss(batch, model, config)

            accuracies[k] = accuracy
            losses[k] = loss.item()

        out[f"{split}_accuracy"] = accuracies.mean()
        out[split] = losses.mean()
        print("Done.")

    model.train()
    return out


def load_dataset_dataloader(
    config,
    cached_data_file_in,
    tokenizer,
    seq_length,
    batch_size,
    num_workers,
    n_gpus,
    offset,
    random_sampling,
):
    # combine multiple dataset files
    if "," in cached_data_file_in:
        cached_data_file_list = cached_data_file_in.split(",")
        cached_data_file_list = [x.strip() for x in cached_data_file_list]
    else:
        cached_data_file_list = [cached_data_file_in]

    dataset_list = []
    dataset_types = []

    for data_file in cached_data_file_list:
        # two dataset types: babylm and tinystories
        dataset_type = "babylm" if "babylm" in data_file else "tinystories"

        dataset_types.append(dataset_type)
        print(
            f"Loading data: {data_file}, \n with seq_length: {seq_length} and batch_size: {batch_size}",
            flush=True,
        )

        # pick correct file automatically, depending on sequence length
        data_file = data_file.format(seq_length=seq_length)

        if config.model_type == "ltg-bert":
            # load dataset for ltg-bert encoder
            dataset = BERTDataset(
                file=data_file,
                offset=offset,
                n_gpus=n_gpus,
                tokenizer=tokenizer,
                seq_length=seq_length,
                mask_p=config.mask_p,
                short_p=config.short_p,
                vocab_size=config.vocab_size,
            )
            print(f"Length Dataset: {len(dataset)}", flush=True)

        else:
            # load dataset for gpt-neo decoder
            dataset = GPTDataset(
                file=data_file,
                offset=offset,
                n_gpus=n_gpus,
                tokenizer=tokenizer,
                seq_length=seq_length,
                vocab_size=config.vocab_size,
            )

        dataset_list.append(dataset)

    if len(dataset_list) > 1:
        dataset = torch.utils.data.ConcatDataset(dataset_list)
        dataset.seq_length = seq_length
        print("Length of the Concatenated Datasets:", len(dataset), flush=True)

        # sample fairly from the two datasets (equal probability)
        if random_sampling:
            # calculate the weights for each dataset
            lens_babylm = 0
            lens_tiny = 0
            for t, d in zip(dataset_types, dataset_list):
                if t == "babylm":
                    lens_babylm += len(d)
                else:
                    lens_tiny += len(d)

            weights = []
            for t, d in zip(dataset_types, dataset_list):
                if t == "babylm":
                    weights += [1 / lens_babylm] * len(d)
                else:
                    weights += [1 / lens_tiny] * len(d)

            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights, num_samples=len(dataset), replacement=True
            )
    else:
        dataset = dataset_list[0]

    if random_sampling and len(dataset_list) > 1:
        print("Sampling fairly")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(config.seed),
            drop_last=True,
            pin_memory=config.pin_memory,
            # should be True for efficiency when using GPU
            # might cause issues with memory
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(config.seed),
            drop_last=True,
            pin_memory=config.pin_memory,
        )

    return dataset, dataloader


def load_data_long_after(
    config,
    steps,
    tokenizer,
    master_process,
    offset,
    n_gpus,
    gradient_accumulation_steps,
):
    print("Entering long after", flush=True)
    # increase sequence length by 4 after some steps
    # need to also decrease batch size for GPU memory limits
    # round to nearest integer
    batch_size_reduced = (config.batch_size // 4 // 4) // 2 * 2
    batch_size_reduced = max(batch_size_reduced, 1)
    batch_size_reduced = 2
    print("Reducing BS ", batch_size_reduced)

    gradient_accumulation_steps = gradient_accumulation_steps * 2

    dataset, dataloader = load_dataset_dataloader(
        config,
        config.cached_data_file,
        tokenizer,
        seq_length=config.seq_length * 4,
        batch_size=batch_size_reduced,
        num_workers=config.num_workers,
        n_gpus=n_gpus,
        offset=offset,
        random_sampling=config.random_sampling,
    )

    if master_process:
        dataset_val, dataloader_val = load_dataset_dataloader(
            config,
            config.cached_valid_data_file,
            tokenizer,
            seq_length=config.seq_length * 4,
            batch_size=batch_size_reduced,
            num_workers=config.num_workers,
            n_gpus=1,
            offset=0,
            random_sampling=config.random_sampling,
        )

    return dataset, dataloader, dataset_val, dataloader_val, gradient_accumulation_steps


def calculate_loss(batch, model, config):
    if config.model_type == "ltg-bert":
        input_ids, attention_mask, target_ids = batch

        attention_mask = attention_mask.to(config.device)
        input_ids = input_ids.to(config.device).t()
        target_ids = target_ids.to(config.device).t()

        prediction = model(input_ids, attention_mask, masked_lm_labels=target_ids)
        if config.DEBUG:
            print("prediction dtype", prediction.dtype)

        # use -100 default cross_entropy ignore value
        # here we filter it out anyway
        target_ids = target_ids.flatten()
        target_ids = target_ids[target_ids != -100]
        loss = F.cross_entropy(prediction, target_ids)

        # no need to exclude with mask
        with torch.no_grad():
            accuracy = (prediction.argmax(-1) == target_ids).float().mean()

        return loss, accuracy

    # loss calculation for gpt-neo
    else:
        input_ids, attention_mask, target_ids = batch

        input_ids = input_ids.to(config.device)
        target_ids = target_ids.to(config.device)
        attention_mask = attention_mask.to(config.device)
        prediction = model(input_ids, attention_mask=attention_mask, labels=target_ids)
        loss, logits, past_key_values = (
            prediction["loss"],
            prediction["logits"],
            prediction["past_key_values"],
        )

        with torch.no_grad():
            accuracy = (logits.argmax(-1) == target_ids).float()
            accuracy[attention_mask == 0] = 0.0
            accuracy = accuracy.mean()

        return loss, accuracy


def prepare_model(config):
    # load encoder/decoder model
    if config.model_type == "ltg-bert":
        model_config = LTGBERTConfig(config_file=config.model_config_file)
        model = LTGBERT(model_config)

    else:
        model_config = AutoConfig.from_pretrained("roneneldan/TinyStories-33M")
        with open(config.model_config_file, "r", encoding="utf-8") as f:
            update_dict = yaml.safe_load(f)

        # update model config values with custom config
        for key, value in update_dict.items():
            setattr(model_config, key, value)
        model = AutoModelForCausalLM.from_config(model_config)
        assert model_config.vocab_size >= config.vocab_size, "Vocab size mismatch!"

    # save it as a dictionary
    model_config = model_config.__dict__

    return model, model_config


def get_model_summary(model, config):
    if config.model_type == "ltg-bert":
        summary(model, [(128, 1), (1, 128)], dtypes=[torch.long, torch.bool])
    else:
        summary(model, [(128, 1)], dtypes=[torch.long])

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total params: {pytorch_total_params}")
    print(f"Total params trainable: {pytorch_total_params_train}")
    # small discrepancy due to shared weights

    return pytorch_total_params


def get_optimizer_and_scheduler(config, model):
    # weight decay specific parameters
    no_decay = ["bias", "layer_norm", "embedding"]
    decay_params = [
        (n, p)
        for n, p in model.named_parameters()
        if not any(nd in n for nd in no_decay)
    ]
    no_decay_params = [
        (n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
    ]
    optimizer_grouped_parameters = [
        {"params": [p for _, p in decay_params], "weight_decay": config.weight_decay},
        {"params": [p for _, p in no_decay_params], "weight_decay": 0.0},
    ]

    # choose optimizer
    # only have adamW for now
    if config.optimizer == "adamW":
        optimizer = torch.optim.AdamW(
            params=optimizer_grouped_parameters,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
        )
    else:
        raise ValueError("Optimizer not supported")

    if config.lr_schedule == "cosine":
        scheduler = LrScheduler(
            max_lr=config.lr,  # max lr, increase lineary
            # for warmup_steps
            warmup_steps=config.warmup_steps_proportion * config.max_steps,
            min_lr=config.min_lr,  # min_lr, decrease until value
            # max_steps for the cosine decay
            max_steps=config.max_steps,
        )

    else:
        # dummy scheduler
        scheduler = ConstScheduler(lr=config.lr)

    return optimizer, scheduler


def setup_optimization_params(config):
    # mixed precision
    # defaults adapted from nanoGPT
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = (
        "cuda" if "cuda" in str(config.device) else "cpu"
    )  # for later use in torch.autocast
    if config.LEGACY:
        ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    else:
        ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=torch.float16)
        )

    if not config.mixed_precision:
        ctx = nullcontext()

    scaler_enabled = config.mixed_precision
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    return scaler, ctx


def seed_everything(seed_value=42):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)


def monitor_CUDA_memory():
    # Get total memory in bytes
    total_memory = torch.cuda.get_device_properties(0).total_memory

    # Get allocated memory (in bytes)
    allocated_memory = torch.cuda.memory_allocated()

    # Get cached memory (in bytes)
    cached_memory = torch.cuda.memory_reserved()

    print(f"Total GPU memory: {total_memory / 1024**2:.2f} MB")
    print(f"Allocated GPU memory: {allocated_memory / 1024**2:.2f} MB")
    print(f"Cached GPU memory: {cached_memory / 1024**2:.2f} MB")


def monitor_CPU_memory():
    process = psutil.Process()
    memory_info = process.memory_info().rss  # Memory used by the main process
    for child in process.children(recursive=True):
        memory_info += child.memory_info().rss  # Add memory used by child processes

    memory_in_mb = memory_info / (1024**2)
    print(f"Total memory used (including children): {memory_in_mb:.2f} MB")
    return memory_in_mb


def get_gradients(model):
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.data.norm(2).item()
    return gradients


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm
