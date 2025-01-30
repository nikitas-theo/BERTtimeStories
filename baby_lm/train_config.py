from types import SimpleNamespace

import yaml
from jsonargparse import ArgumentParser

# add your own, alternatively the `wandb_log` option can be set to false
WANDB_KEY = None

# use the HF implementation for the LTG-BERT model, else use the implementation in `baby_lm/encoder/`
# the models were trained with USE_HF = False
USE_HF = False


def get_config():
    """
    this config setup assumes no default values in code, and they are only set in the yaml config files
    --training_config is the base config file
    --experiment_config updates and overrides the training_config options

    finally command line arguments have the highest priority and override the previous two

    this creates a hierarchy:
        command_line > experiment_config > training_config
    """

    def update_args(from_config, to_config, from_cmd):
        """
        update `to_config` with `from_config`
        if `from_cmd` is True, only update if value is not None
        """
        # iterate over input Namespace
        for key, value in from_config.__dict__.items():
            # if from_cmd is True, only update if value is not None
            if hasattr(to_config, key) and from_cmd:
                if value is not None:
                    setattr(to_config, key, value)
            # else just update (even if it is None)
            else:
                setattr(to_config, key, value)
        return to_config

    parser = ArgumentParser()

    # load yaml config file
    parser.add_argument("--training_config", type=str, help="base config file")
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="experiment config file (overrides training_config)",
    )
    parser.add_argument("--wandb_project", type=str, help="wandb project name")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["ltg-bert", "gpt"],
        help="model type, ltg-bert or gpt",
    )
    parser.add_argument("--device", type=str, help="training device")
    parser.add_argument("--model_config_file", type=str, help="model config file")
    parser.add_argument(
        "--cached_data_file",
        type=str,
        help="cached data file, \
        created by the data preprocessing script, multiple files can be provided separated with ','",
    )
    parser.add_argument(
        "--cached_valid_data_file",
        type=str,
        help="cached valid data file, \
        created by the data preprocessing script",
    )
    parser.add_argument("--tokenizer_path", type=str, help="tokenizer path")
    parser.add_argument(
        "--seq_length",
        type=int,
        help="sequence length, corresponding cached data file should exist",
    )
    parser.add_argument("--batch_size", type=int, help="batch size")

    ######################
    # one of them should be defined
    parser.add_argument("--max_steps", type=int, help="number of training steps")
    parser.add_argument("--epochs", type=int, help="number of training epochs")
    ######################

    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--optimizer", type=str, help="optimizer, only adamW for now")
    parser.add_argument("--betas", type=float, nargs=2, help="beta values for adamW")
    parser.add_argument("--eps", type=float, help="epsilon value for adamW")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--weight_decay", type=float, help="weight decay")
    parser.add_argument("--grad_clip", type=float, help="gradient clipping value")

    parser.add_argument(
        "--output_dir", type=str, help="output directory for logs, checkpoints, etc."
    )

    parser.add_argument(
        "--load_checkpoint",
        type=str,
        help="checkpoint file to load for resuming training, if None start a new training",
    )

    parser.add_argument(
        "--lr_schedule",
        type=str,
        help="learning rate scheduler, only const or cosine for now",
    )
    parser.add_argument(
        "--max_lr", type=float, help="max learning rate for the scheduler"
    )
    parser.add_argument(
        "--min_lr", type=float, help="min learning rate for the scheduler"
    )
    parser.add_argument(
        "--warmup_steps_proportion",
        type=float,
        help="proportion of max_steps to do lr warmup",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument("--compile", type=bool, help="compile the model, if supported")

    parser.add_argument(
        "--eval_every",
        type=int,
        help="evaluate model performance every `eval_every` steps",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        help="number of iterations of the validation data used to compute validation loss",
    )
    parser.add_argument(
        "--print_every", type=int, help="print training stats every `print_every` steps"
    )
    parser.add_argument(
        "--long_after",
        type=float,
        help="percentage of max steps to switch to longer sequence length\
                        e.g., after 0.9 percent of max steps, switch to 512",
    )
    parser.add_argument("--wandb_log", type=bool, help="log to wandb optionally")
    parser.add_argument("--mask_p", type=float, help="masking probability for ltg-bert")
    parser.add_argument(
        "--short_p", type=float, help="probability used in span masking"
    )
    parser.add_argument(
        "--always_save_checkpoint",
        type=bool,
        help="always save the checkpoint, even if it is not the best",
    )
    parser.add_argument("--wandb_offline", type=bool, help="wandb offline mode")
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="vocab size, should be the same as in model config file",
    )

    parser.add_argument(
        "--LEGACY",
        type=bool,
        help="legacy mode, for compatibility with servers with older codebases, \
                        added for the mixed precision training",
    )
    parser.add_argument("--DEBUG", type=bool, help="debug mode, for debugging purposes")

    parser.add_argument("--run_name", type=str)
    parser.add_argument(
        "--wandb_id",
        type=str,
        help="wandb run id, for resuming training. This is set automatically.",
    )
    parser.add_argument("--save_dir", type=str, help="this is set automatically.")

    parser.add_argument(
        "--num_workers", type=int, help="number of workers for the dataloader"
    )
    parser.add_argument("--pin_memory", type=bool, help="pin memory for the dataloader")

    parser.add_argument("--mixed_precision", type=bool, help="mixed precision training")

    parser.add_argument(
        "--random_sampling",
        type=bool,
        help="i.e., balanced training, random sampling for the data loader",
    )

    config_cmd = parser.parse_args()

    # hacky way to define a hierarchy of config files
    with open(config_cmd.training_config, "r", encoding="utf-8") as f:
        training_config = SimpleNamespace(**yaml.safe_load(f))

    with open(config_cmd.experiment_config, "r", encoding="utf-8") as f:
        experiment_config = SimpleNamespace(**yaml.safe_load(f))

    config_pre_defined = update_args(
        from_config=experiment_config, to_config=training_config, from_cmd=False
    )

    config_pre_defined_and_cmd = update_args(
        from_config=config_cmd, to_config=config_pre_defined, from_cmd=True
    )

    config = config_pre_defined_and_cmd

    for key, _ in config.__dict__.items():
        assert hasattr(config_cmd, key), "key {} not found in config_cmd".format(key)

    # hacky way so we can set None in command line
    if config.max_steps == -1:
        config.max_steps = None

    if config.epochs == -1:
        config.epochs = None

    assert config.max_steps is None or config.max_steps >= 0
    assert config.epochs is None or config.epochs >= 0

    assert (
        config.epochs is None or config.max_steps is None
    ), "Either `epochs` or `max_steps` should be defined, but not both"
    assert (
        config.epochs is not None or config.max_steps is not None
    ), "One of `epochs` or `max_steps` should be defined"

    return config
