import os

import jsonargparse

from baby_lm.process_data._prepare_utils import (
    run_command,
    sample_dataset,
    seed_everything,
)

split_dict = {
    "train": "TinyStoriesV2-GPT4-train",
    "valid": "TinyStoriesV2-GPT4-valid",
    "instruct_train": "TinyStories-Instruct-train",
    "instruct_valid": "TinyStories-Instruct-valid",
}


def process_tinystories(config, split):
    base_raw = config.base_raw
    num_words = config.num_words
    millions = num_words // 1_000_000
    print(f"Processing TinyStories {split} with {millions}M words")

    sampled_path = f"{config.base_dir}/{split}_{millions}M"

    print("sample dataset")

    # sample e.g., 100M words at the start of processing.
    sample_dataset(
        f"{base_raw}{split}.txt",
        sampled_path + ".txt",
        num_words,
        special="<|endoftext|>",
    )

    print("Processing TinyStories")

    # process the Tiny Stories data
    run_command(
        f"python -m baby_lm.process_data.process_tiny_stories --path {sampled_path}.txt"
    )

    # just add proc and segmented
    run_command(
        f"python -m baby_lm.process_data.segment --filepath {sampled_path}_proc.txt --model_type gpt"
    )


def cache_tinystories(config, split, in_tokenizer_path):
    vocab_size = config.vocab_size
    min_frequency = config.min_frequency
    seq_lens = config.seq_lens

    num_words = config.num_words
    millions = num_words // 1_000_000

    if in_tokenizer_path is not None:
        print("Found  Tokenizer")
        tokenizer_path = in_tokenizer_path
    else:
        print("Training Tokenizer")
        tokenizer_path = f"./outputs/tokenizers/tinystories/{config.split_original}_{millions}M/tokenizer.json"
        # if a tokenizer file is not provided, train a tokenizer
        run_command(
            "python -m baby_lm.create_tokenizer "
            + f"--input_path {config.base_dir}/{split}_{millions}M_proc_segmented.txt "
            + f"--vocab_path {tokenizer_path} "
            + f"--vocab_size {vocab_size} "
            + f"--min_frequency {min_frequency} "
            + "--model_type gpt"
        )

    print("Caching TinyStories")
    for seq_len in seq_lens:
        # now cache the data
        run_command(
            "python -m baby_lm.process_data.cache_tiny_stories "
            + f"--segments_path  {config.base_dir}/{split}_{millions}M_proc_segmented.txt "
            + f"--tokenizer_path {tokenizer_path} "
            + f"--sequence_length {seq_len}"
        )
    return tokenizer_path


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()

    parser.add_argument(
        "--config_file", action="config", help="Path to the configuration file."
    )

    parser.add_argument(
        "--base_raw",
        type=str,
        help="Path to the raw data directory",
        default="./data/raw/tinystories/",
    )

    parser.add_argument(
        "--base_proc",
        type=str,
        help="Path to the processed data directory",
        default="./data/processed/tinystories/",
    )

    # split to sample from
    parser.add_argument("--split", type=str, default="train")

    # words to include from split (excluding <|endoftext|>)
    parser.add_argument("--num_words", type=int, default=None)

    # vocab size for caching
    parser.add_argument("--vocab_size", type=int, default=None)

    # tokenizer stuff
    parser.add_argument("--min_frequency", type=int, default=10)

    # sequence lengths to cache
    parser.add_argument("--seq_lens", type=str, default="512")

    # pre-load tokenizer if necessary (optional)
    parser.add_argument("--tokenizer_path", type=str, default=None)

    config = parser.parse_args()

    # this is for consistent document shuffle (not necessary here)
    seed_everything(42)

    # separate the sequence lengths
    config.seq_lens = [int(x) for x in config.seq_lens.split(",")]

    # create a base dir for this split
    config.base_dir = (
        f"{config.base_proc}{config.split}_{config.num_words // 1_000_000}M/"
    )
    os.makedirs(config.base_dir, exist_ok=True)

    # keep the original shorter name
    config.split_original = config.split

    # translate to the actual name
    config.split = split_dict[config.split]

    # process split
    process_tinystories(config, config.split)
    tokenizer_path = cache_tinystories(config, config.split, config.tokenizer_path)

    ###########################################################
    # config.base_dir and config.split_original will remain the same
    if config.split_original == "train":
        valid = "valid"
    elif config.split_original == "instruct_train":
        valid = "instruct_valid"

    # process the dev independently for each split
    process_tinystories(config, split_dict[valid])
    cache_tinystories(config, split_dict[valid], tokenizer_path)
