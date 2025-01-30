import json
import os

import jsonargparse
import yaml

from baby_lm.process_data._prepare_utils import (
    run_command,
    sample_dataset,
    seed_everything,
)


def main_generated(generated_data_in_path, seq_lens, tokenizer_path):
    print("Processing generated data")
    path = generated_data_in_path

    # need to convert the data to .txt
    ##################################
    basedir = os.path.dirname(path)

    config_generation_file = basedir + "/config.yaml"

    with open(config_generation_file, "r") as f:
        config_generation = yaml.safe_load(f)

    model_name = config_generation["models"]

    with open(path, "r") as f:
        print(path)
        stories_list = json.load(f)

    print("Writing to txt")
    # hopefully we have enough RAM
    # unpack for all possible generations for the same model

    print("removing possible use of _HF in model name for online models")
    model_name = model_name.replace("_HF", "")

    dataset = []
    for story in stories_list:
        for generation in story[model_name]:
            dataset += [story["prompt"] + generation]
    special = "[PAR]"
    dataset_str = f"\n{special}\n".join(dataset)
    path_save = basedir + "/dataset_generated"
    with open(path_save + ".txt", "w") as f:
        f.write(dataset_str)

    ##################################

    print("Processing data")
    # process generated data
    print(path_save)
    run_command(
        f"python -m baby_lm.process_data.process_tiny_stories --path {path_save}.txt"
    )
    path_save = path_save.replace("raw", "processed")
    print("Segmenting data")
    # segment generated data
    run_command(
        f"python -m baby_lm.process_data.segment --filepath {path_save}_proc.txt --model_type gpt"
    )

    # cache generated dataset
    for seq_len in seq_lens:
        print(f"Cache {seq_len}.")
        run_command(
            "python -m baby_lm.process_data.cache_baby_lm "
            + f"--segments_path  {path_save}_proc_segmented.txt "
            + f"--tokenizer_path {tokenizer_path} "
            + f"--sequence_length {seq_len}"
        )

    return f"{path_save}_proc_segmented_cached_*.txt"


def process_val(config, tokenizer_path):
    val_path = config.val_path
    for seq_len in config.seq_lens:
        print(f"Cache {seq_len}.")
        run_command(
            "python -m baby_lm.process_data.cache_baby_lm "
            + f"--segments_path  {val_path} "
            + f"--tokenizer_path {tokenizer_path} "
            + f"--sequence_length {seq_len}"
        )
    base_path, _ = os.path.splitext(val_path)
    return base_path + "_cached_"


def main(config):
    vocab_size = config.vocab_size
    min_frequency = config.min_frequency

    millions_babylm = config.millions_babylm
    millions_tiny = config.millions_tiny

    base_proc = f"./data/processed/joint/joint_{millions_babylm}M_{millions_tiny}M_{config.name_extra}/"

    os.makedirs(base_proc, exist_ok=True)

    # sample BabyLM dataset number of words (from processed)
    if config.sample_babylm is not None:
        print("Sampling first")
        path_out_babylm = f"{base_proc}/dataset_babylm_sampled_{millions_babylm}M.txt"
        sample_dataset(
            config.babylm_path,
            path_out_babylm,
            config.sample_babylm,
            shuffle_documents=True,
        )
    else:
        print("skip sampling")
        path_out_babylm = config.babylm_path

    # sample TinyStories dataset number of words (from processed)
    if config.sample_tinystories is not None:
        print("Sampling second")

        path_out_tinystories = (
            f"{base_proc}/dataset_tinystories_sampled_{millions_tiny}M.txt"
        )
        sample_dataset(
            config.tinystories_path,
            path_out_tinystories,
            config.sample_tinystories,
            special="<|endoftext|>",
        )
    else:
        print("skip sampling")
        path_out_tinystories = config.tinystories_path

    ####################
    print("Replacing special token <|endoftext|> with [PAR]")
    with open(path_out_tinystories, "r") as file:
        filedata = file.read()

    filedata = filedata.replace("<|endoftext|>", "[PAR]")
    if config.sample_tinystories is not None:
        path_out_tinystories_replaced = path_out_tinystories.replace(
            ".txt", "_with_PAR.txt"
        )
    else:
        path_out_tinystories_replaced = (
            f"{base_proc}/dataset_tinystories_{millions_tiny}M_with_PAR.txt"
        )

    with open(path_out_tinystories_replaced, "w") as file:
        file.write(filedata)

    # replace the path
    path_out_tinystories = path_out_tinystories_replaced
    ######################

    print("Concatenating")
    # concatenate
    # cat will add \n automatically between files
    run_command(
        f"cat {path_out_babylm} {path_out_tinystories} > {base_proc}/processed_segmented_cat.txt"
    )

    tokenizer_path = f"./outputs/tokenizers/joint/joint_{millions_babylm}M_{millions_tiny}M_{config.name_extra}/tokenizer.json"
    segments_path = f"{base_proc}/processed_segmented_cat.txt"

    # train tokenizer
    print("train tokenizer")
    run_command(
        "python -m baby_lm.create_tokenizer "
        + f"--input_path {segments_path} "
        + f"--vocab_path {tokenizer_path} "
        + f"--vocab_size {vocab_size} "
        + f"--min_frequency {min_frequency} "
        + "--model_type ltg-bert"
    )

    # cache dataset
    for seq_len in config.seq_lens:
        # process separately for more flexibility
        def fix_path(path):
            base_directory = os.path.dirname(path)
            file_name = os.path.basename(path)
            if base_directory != base_proc:
                os.system(f"mv {path} {base_proc}/{file_name}")

            return f"{base_proc}/{file_name}"

        path_out_babylm = fix_path(path_out_babylm)
        path_out_tinystories = fix_path(path_out_tinystories)

        print(f"Cache {seq_len}.")
        run_command(
            "python -m baby_lm.process_data.cache_baby_lm "
            + f"--segments_path  {path_out_babylm} "
            + f"--tokenizer_path {tokenizer_path} "
            + f"--sequence_length {seq_len}"
        )
        run_command(
            "python -m baby_lm.process_data.cache_baby_lm "
            + f"--segments_path  {path_out_tinystories} "
            + f"--tokenizer_path {tokenizer_path} "
            + f"--sequence_length {seq_len}"
        )

    return base_proc, tokenizer_path


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()

    parser.add_argument(
        "--config_file", action="config", help="Path to the configuration file."
    )

    # sample from babylm
    parser.add_argument("--babylm_path", type=str)
    parser.add_argument("--sample_babylm", type=int)
    parser.add_argument("--millions_babylm", type=int)

    # sample from tinystories
    parser.add_argument("--tinystories_path", type=str)
    parser.add_argument("--sample_tinystories", type=int)
    parser.add_argument("--millions_tiny", type=int)

    # use generated data
    parser.add_argument("--generated_data_in_path", type=str)
    parser.add_argument("--val_path", type=str)

    # tokenizer
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--min_frequency", type=int, default=10)
    parser.add_argument("--seq_lens", type=str, default="128")

    # e.g., differentiate between sampling strategies
    parser.add_argument("--name_extra", type=str, default="")

    # process validation data
    parser.add_argument("--do_validation", type=bool, default=True)

    config = parser.parse_args()

    do_validation = config.do_validation

    # if a generated data path is not specified we skip it
    do_generated = config.generated_data_in_path is not None

    # for sampling documents
    seed_everything(42)

    config.seq_lens = [int(x) for x in config.seq_lens.split(",")]

    ########################################
    # process main datasets
    base_proc, tokenizer_path = main(config)

    # process generated data
    if do_generated:
        # process generated data
        path_generated = main_generated(
            config.generated_data_in_path, config.seq_lens, tokenizer_path
        )
        # take the relevant generated data that we have cached with our tokenizer
        run_command(f"mv {path_generated} {base_proc}/")

    # process validation data
    if do_validation:
        # process validation data
        # validation is just babylm so its ok.
        final_path = process_val(config, tokenizer_path)
        for seq in config.seq_lens:
            run_command(
                f"mv {final_path}{seq}.txt "
                f"{base_proc}/DEV_all_segmented_cached_{seq}.txt"
            )
