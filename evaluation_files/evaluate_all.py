import os
import subprocess

import yaml

#  this parameter doesn't do anything
# and everything is evaluated with BS=1
BLIMP_EWOK_BATCH_SIZE = 512
SEED = 42


# Function to execute shell commands
def run_command(cmd):
    print(cmd)
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()


def run_command_error_handling(cmd):
    print(f"Running command: {' '.join(cmd)}")
    try:
        # Set stdout and stderr to None for real-time display
        result = subprocess.run(cmd, check=True, text=True, stdout=None, stderr=None)
        print("Command succeeded.")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        exit(1)


def finetune_GLUE(model_path, model_name, max_len, glue_config_file):
    tasks = [
        "boolq",
        "cola",
        "mnli",
        "mnli-mm",
        "mrpc",
        "multirc",
        "qnli",
        "qqp",
        "rte",
        "sst2",
        "wsc",
    ]
    batch_size = {
        "boolq": 32,  # 2
        "mnli-mm": 32,  # 2
        "mnli": 32,  # 2
        "mrpc": 32,  # 2
        "multirc": 32,  #####
        "qnli": 32,  # 2
        "qqp": 32,  # 2
        "sst2": 32,  # single
        "cola": 16,  # single
        "rte": 16,  # 2
        "wsc": 16,  # 2
    }
    for task in tasks:
        print(f"Finetuning {model_name} on {task}")
        # extra case for mnli-mm
        # we use a pre-trained model
        if task == "mnli-mm":
            train_name = "mnli"
            valid_name = "mnli-mm"
            do_train = False
            model_path_full = f"results/finetune/{model_name}/{train_name}/"
        else:
            train_name = task
            valid_name = task
            do_train = True
            model_path_full = model_path

        os.makedirs(f"results/finetune/{model_name}/{task}/", exist_ok=True)
        with open(glue_config_file, "r") as f:
            config = yaml.safe_load(f)

        max_epochs = config["num_epochs"]

        # dropout is a model configuration parameter, no need to set it here
        # dropout = config['dropout']

        warmup_proportion = config["warmup_proportion"]
        lr = config["learning_rate"]
        decay_type = config["decay_type"]
        weight_decay = config["weight_decay"]
        optim = config["optim"]
        adam_epsilon = config["adam_epsilon"]
        adam_beta1 = config["adam_beta1"]
        adam_beta2 = config["adam_beta2"]
        bsz = batch_size[task]
        # patience = config['patience']
        # max_length = config['max_length']
        max_length = max_len
        seed = SEED

        command = [
            "python",
            "finetune_classification.py",
            "--model_name_or_path",
            model_path_full,
            "--output_dir",
            f"results/finetune/{model_name}/{task}/",
            "--train_file",
            f"evaluation_data/glue_filtered/{train_name}.train.jsonl",
            "--validation_file",
            f"evaluation_data/glue_filtered/{valid_name}.valid.jsonl",
            "--do_train",
            str(do_train),
            "--do_eval",
            "--do_predict",
            "--max_seq_length",
            str(max_length),
            "--per_device_train_batch_size",
            str(bsz),
            "--learning_rate",
            str(lr),
            "--num_train_epochs",
            str(max_epochs),
            # "--patience", str(patience),
            "--evaluation_strategy",
            "epoch",
            "--save_strategy",
            "epoch",
            "--overwrite_output_dir",
            "--seed",
            str(seed),
            "--adam_epsilon",
            adam_epsilon,
            "--adam_beta1",
            adam_beta1,
            "--adam_beta2",
            adam_beta2,
            "--optim",
            optim,
            "--weight_decay",
            weight_decay,
            "--lr_scheduler_type",
            decay_type,
            "--warmup_ratio",
            warmup_proportion,
            "--max_grad_norm",
            2.0,
            "--trust_remote_code",
            "--save_total_limit",
            1,
            "--load_best_model_at_end",
            True,
            "--fp16",
            True,
        ]
        command = [str(c) for c in command]
        print(command)
        os.system(
            " ".join(command) + f" | tee results/finetune/{model_name}/{task}/tee.log"
        )


def eval_ewok(
    model_type,
    model_name,
    model_path,
    batch_size=BLIMP_EWOK_BATCH_SIZE,
    device="cuda",
):
    if model_type == "mlm":
        model = "hf-mlm"
        model_args = f"pretrained={model_path},backend=mlm"
    else:
        model = "hf"
        model_args = f"pretrained={model_path}"

    output_dir = f"results/ewok/{model_name}/"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "lm_eval",
        "--model",
        model,
        "--model_args",
        model_args,
        "--tasks",
        "ewok_filtered",
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--output_path",
        output_dir + "ewok_results.json",
        "--trust_remote_code",
        "--log_samples",
    ]

    command = [str(c) for c in cmd]

    print(command)
    os.system(" ".join(command) + f" | tee results/ewok/{model_name}/tee.log")


def eval_blimp(
    model_type,
    model_name,
    model_path,
    batch_size=BLIMP_EWOK_BATCH_SIZE,
    device="cuda",
    tasks="blimp_filtered,blimp_supplement",
):
    if model_type == "mlm":
        model = "hf-mlm"
        model_args = f"pretrained={model_path},backend=mlm"
    else:
        model = "hf"
        model_args = f"pretrained={model_path}"

    output_dir = f"results/blimp/{model_name}/"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "lm_eval",
        "--model",
        model,
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--output_path",
        output_dir + "blimp_results.json",
        "--trust_remote_code",
        "--log_samples",
    ]

    command = [str(c) for c in cmd]
    print(command)
    os.system(" ".join(command) + f" | tee results/blimp/{model_name}/tee.log")


def run_evaluation(eval_config, max_len, eval_tasks):
    # comment out corresponding lines if you don't want to evaluate
    # on a specific benchmark

    for model_config in eval_config:
        path = model_config["model_path"]
        model_name = model_config["model_name"]
        model_type = model_config["model_type"]
        if "blimp" in eval_tasks:
            eval_blimp(model_type, model_name, path)
        if "ewok" in eval_tasks:
            eval_ewok(model_type, model_name, path)

    for model_config in eval_config:
        path = model_config["model_path"]
        model_name = model_config["model_name"]
        if "glue" in eval_tasks:
            finetune_GLUE(path, model_name, max_len)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glue_config_file", type=str, default="ltg_bert_glue_config_finetune.yaml"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="use an evaluation config containing model name, path and type for a list of models",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="if config is not defined, use this model name",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="if config is not defined, use this model path",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="if config is not defined, use this model type",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="maximum sequence length for finetuning",
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default="blimp, ewok, glue",
        help="comma separated list of tasks to evaluate",
    )
    args = parser.parse_args()

    if args.config_file is not None:
        assert args.model_name is None
        assert args.model_path is None
        assert args.model_type is None
        with open(args.config_file, "r") as f:
            eval_config = yaml.safe_load(f)

    if args.model_name is not None:
        assert args.model_path is not None
        assert args.model_type is not None
        assert args.config_file is None
        eval_config = [
            {
                "model_name": args.model_name,
                "model_path": args.model_path,
                "model_type": args.model_type,
            }
        ]

    if args.config_file is None and args.model_name is None:
        raise ValueError(
            "Please provide either a config file or a model name, path and type."
        )

    print(args.max_len)
    args.eval_tasks = args.eval_tasks.split(",")
    run_evaluation(eval_config, args.max_len, args.eval_tasks)
