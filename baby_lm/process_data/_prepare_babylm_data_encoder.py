import jsonargparse

from baby_lm.process_data._prepare_utils import run_command, seed_everything


def process_babylm(config, split):
    base_proc = config.base_proc

    print("Pre-processing split:", split)
    # first pre-process the specific split
    run_command(
        f"python -m baby_lm.process_data.process_baby_lm --dataset all --split {split}"
    )

    if split == "dev":
        suffix = "dev"
    else:
        suffix = "train"

    print("Concatenating files")
    # then do a concatenation of all sub-dataset files
    run_command(f"cat {base_proc}/{split}/*.{suffix} > {base_proc}/{split}/all.txt")

    print("Segmenting concatenated file")
    # then segment the concatenated file into sentences
    run_command(
        f"python -m baby_lm.process_data.segment --filepath {base_proc}/{split}/all.txt --model_type bert"
    )


def cache_babylm(config, split, in_tokenizer_path):
    base_proc = config.base_proc
    vocab_size = config.vocab_size
    min_frequency = config.min_frequency
    seq_lens = config.seq_lens

    if in_tokenizer_path is not None:
        print("Found  Tokenizer")
        tokenizer_path = in_tokenizer_path
    else:
        print("Training Tokenizer")
        tokenizer_path = f"./outputs/tokenizers/babylm/{split}/tokenizer.json"
        # if a tokenizer file is not provided, train a tokenizer
        run_command(
            "python -m baby_lm.create_tokenizer "
            + f"--input_path {base_proc}/{split}/all_segmented.txt "
            + f"--vocab_path {tokenizer_path} "
            + f"--vocab_size {vocab_size} "
            + f"--min_frequency {min_frequency} "
            + "--model_type ltg-bert"
        )
    for seq_len in seq_lens:
        # now cache the data
        run_command(
            "python -m baby_lm.process_data.cache_baby_lm "
            + f"--segments_path {base_proc}/{split}/all_segmented.txt "
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
        "--base_proc",
        type=str,
        default="data/processed/babylm/",
        help="base directory for processed data",
    )
    parser.add_argument(
        "--split", type=str, default="train_100M", help="babylm split to use"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=16384, help="vocab size for the tokenizer"
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=10,
        help="minimum frequency for the tokenizer",
    )
    parser.add_argument(
        "--seq_lens",
        type=str,
        default="128",
        help="comma separated list of sequence lengths to cache",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="path to tokenizer file, if None will be created",
    )

    config = parser.parse_args()
    seed_everything(42)
    config.seq_lens = [int(x) for x in config.seq_lens.split(",")]

    process_babylm(config, config.split)
    tokenizer_path = cache_babylm(config, config.split, config.tokenizer_path)

    # process the dev independently for each split
    process_babylm(config, "dev")
    cache_babylm(config, "dev", tokenizer_path)
    for seq in config.seq_lens:
        run_command(
            f"mv {config.base_proc}/dev/all_segmented_cached_{seq}.txt "
            f"{config.base_proc}/{config.split}/DEV_all_segmented_cached_{seq}.txt"
        )
