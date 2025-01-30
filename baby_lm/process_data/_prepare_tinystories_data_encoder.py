import jsonargparse

from baby_lm.process_data._prepare_joint_training_data_encoder import main_generated
from baby_lm.process_data._prepare_utils import run_command


def replace_special_token(path_tiny_stories):
    print("Replacing special token <|endoftext|> with [PAR]")
    with open(path_tiny_stories, "r") as file:
        filedata = file.read()

    filedata = filedata.replace("<|endoftext|>", "[PAR]")
    path_out_tinystories_replaced = path_tiny_stories.replace(".txt", "_with_PAR.txt")
    with open(path_out_tinystories_replaced, "w") as file:
        file.write(filedata)

    return path_out_tinystories_replaced


def create_tinystories_ltg_bert(args):
    millions_tiny = args.millions_tiny
    vocab_size = args.vocab_size
    min_frequency = args.min_frequency

    # get train and valid, and replace
    path_tiny_stories_train = f"./data/processed/tinystories/train_{millions_tiny}M/TinyStoriesV2-GPT4-train_{millions_tiny}M_proc_segmented.txt"
    path_tiny_stories_valid = f"./data/processed/tinystories/train_{millions_tiny}M/TinyStoriesV2-GPT4-valid_{millions_tiny}M_proc_segmented.txt"
    path_out_tinystories_replaced_train = replace_special_token(path_tiny_stories_train)
    path_out_tinystories_replaced_valid = replace_special_token(path_tiny_stories_valid)

    base_proc = f"./data/processed/tinystories/train_{millions_tiny}M/"

    print(f"Path replaced train: {path_out_tinystories_replaced_train}")
    print(f"Path replaced valid: {path_out_tinystories_replaced_valid}")

    # we store it in the same directory, but different ending
    tokenizer_path = f"./outputs/tokenizers/tinystories/train_{millions_tiny}M/tokenizer_ltg_bert.json"
    print(f"Creating tokenizer at {tokenizer_path}")

    # create tokenizer
    run_command(
        "python -m baby_lm.create_tokenizer "
        + f"--input_path {path_out_tinystories_replaced_train} "
        + f"--vocab_path {tokenizer_path} "
        + f"--vocab_size {vocab_size} "
        + f"--min_frequency {min_frequency} "
        + "--model_type ltg-bert"
    )
    print(f"Path train {path_tiny_stories_train}")
    print(f"Path valid {path_tiny_stories_valid}")

    for seq_len in args.seq_lens:
        # cache train
        run_command(
            "python -m baby_lm.process_data.cache_baby_lm "
            + f"--segments_path {path_out_tinystories_replaced_train} "
            + f"--tokenizer_path {tokenizer_path} "
            + f"--sequence_length {seq_len}"
        )
        # cache valid
        run_command(
            "python -m baby_lm.process_data.cache_baby_lm "
            + f"--segments_path {path_out_tinystories_replaced_valid} "
            + f"--tokenizer_path {tokenizer_path} "
            + f"--sequence_length {seq_len}"
        )
    return tokenizer_path, base_proc


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        "--config_file", action="config", help="Path to the configuration file."
    )

    # the split to prepare
    parser.add_argument("--millions_tiny", type=int, required=True)

    # the vocab size
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--min_frequency", type=int, required=True, default=10)

    # ltg-bert training, 128
    parser.add_argument("--seq_lens", type=str, default="128")

    # also process generated data
    parser.add_argument("--generated_data_path", type=str, default=None)

    args = parser.parse_args()
    args.seq_lens = args.seq_lens.split(",")
    tokenizer_path, base_proc = create_tinystories_ltg_bert(args)
    if args.generated_data_path is not None:
        path_generated = main_generated(
            args.generated_data_path, args.seq_lens, tokenizer_path
        )
        run_command(f"mv {path_generated} {base_proc}/")
