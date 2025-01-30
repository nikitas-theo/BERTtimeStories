import datetime
import json
import os
import random

import numpy as np
import torch
import yaml
from jsonargparse import ArgumentParser
from tqdm import tqdm

from baby_lm.generator.sample_utils import generate_sequences

SAVE_PATH = "./data/raw/generated/"
MODELS_PATH = "./outputs/models/"


def seed_everything(seed_value):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)


def read_train(PATH):
    with open(PATH, "r") as f:
        num_words = 0
        lines = ""
        for line in tqdm(f):
            num_words += len(line.split())
            lines += " " + line.strip()
    print(f"Loaded {num_words} words")
    return lines


def sampled_from_dataset(data_path, num_stories):
    """
    sample a number of `num_stories` stories from a dataset
    """
    lines = read_train(data_path)
    stories = lines.split("<|endoftext|>")
    stories = [story.strip() for story in stories]
    stories = [story for story in stories if len(story) > 0]

    random.shuffle(stories)
    print(f"Found {len(stories)} stories")

    if num_stories is None:
        sampled = stories
    else:
        sampled = stories[0:num_stories]

    return sampled


def sample_from_file(filepath, num_stories):
    """
    sample a number of `num_stories` stories from a file
    """

    with open(filepath, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    prompts = [p.replace("“", '"').replace("’", "'").replace("”", '"') for p in prompts]
    prompts = [p.replace("\n", " ") for p in prompts]

    random.shuffle(prompts)
    if num_stories is None:
        pass
    else:
        prompts = prompts[0:num_stories]

    return prompts


def get_stories(config, models):
    is_prompt = config.is_prompt
    if is_prompt:
        print("sampling from prompts file")
        sampled = sample_from_file(config.sample_from, config.num_stories)
    else:
        sampled = sampled_from_dataset(config.sample_from, config.num_stories)

    # create prompts, and the input to the model
    inputs = []
    prompts = []
    outputs = [{} for _ in range(len(sampled))]

    big_stories_cnt = 0
    small_stories_cnt = 0
    for i, story in enumerate(sampled):
        # if it's a prompt, we don't need to cut it
        if not is_prompt:
            len_story = len(story.split())

            # cut long stories
            if len_story > 512:
                print(f"Warning: story {i} is too long, cutting it")
                len_story = 512
                big_stories_cnt += 1

            # skip small stories
            if len_story < 20:
                print(f"Warning: story {i} is too short, skipping it")
                small_stories_cnt += 1
                continue

            # keep a random starting length as a prompt
            len_keep = np.random.randint(
                int(config.prob_min * len_story), int(config.prob_max * len_story)
            )

            # story accepted
            inputs.append({})
            prompt = " ".join(story.split()[:len_keep])
            inputs[-1]["prompt"] = prompt
            inputs[-1]["complete_orig"] = " ".join(story.split()[len_keep:])

        else:
            # story accepted
            inputs.append({})
            prompt = story
            inputs[-1]["prompt"] = prompt
            inputs[-1]["complete_orig"] = ""

        inputs[-1]["story"] = story
        prompts.append(prompt)

    print("Number of big stories:", big_stories_cnt)
    print("Number of small stories:", small_stories_cnt)
    print("Number of surviving stories after filtering:", len(prompts))
    print("constructed prompts")

    outputs = [{} for _ in range(len(prompts))]

    # for each model, generate
    for name in models:
        # HF model, e.g., roneneldan/TinyStories-33M_HF
        if "_HF" in name:
            name = name.replace("_HF", "")
            HF = True
            model_load_path = name

        # else, load from baby_lm/outputs/models/.../
        else:
            HF = False
            model_load_path = MODELS_PATH + name + "/checkpoint/"

        # copy generation parameters
        eval_keys_filtered = [
            x.replace("eval_", "") for x in vars(config).keys() if x.startswith("eval_")
        ]

        generation_parameters = {
            key: getattr(config, "eval_" + key) for key in eval_keys_filtered
        }

        with torch.no_grad():
            # use generation pipeline
            generations = generate_sequences(
                prompts,
                model_load_path=model_load_path,
                generation_parameters_dict=generation_parameters,
                HF=HF,
            )
        outputs = [
            {**outputs[i], **generations[i], **inputs[i]}
            for i in range(len(generations))
        ]

        # for older python version
        # outputs = [
        #    {**outputs[i], **generations[i], **inputs[i]}
        #     for i in range(len(sampled))
        # ]

        for i in range(len(outputs)):
            del outputs[i]["generated_text"]

        for i in range(len(outputs)):
            outputs[i][name] = outputs[i]["model_completion"]
            del outputs[i]["model_completion"]

    return outputs


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--config_file",
        action="config",
        help="path to the configuration file used for the sampling",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="""
            models to use for generation, comma separated list, 
            HF models supported with syntax roneneldan/TinyStories-33M_HF,
            baby_lm models supported by just giving the name in /outputs/models/name
        """,
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--folder_name",
        type=str,
        default=f"gen_{datetime.datetime.now()}",
        help="name of the folder to save the generated files in outputs/generated_data/",
    )

    # story sampling for prompt creation
    parser.add_argument(
        "--num_stories",
        type=int,
        default=None,
        help="number of stories to sample from the dataset for prompt creation",
    )
    parser.add_argument(
        "--sample_from",
        type=str,
        default=None,
        help="path to dataset to sample from, or prompt",
    )
    parser.add_argument(
        "--is_prompt",
        type=bool,
        default=False,
        help="indicate whether this is a dataset or a prompt",
    )

    # cutoff story for prompt-making based on length
    parser.add_argument(
        "--prob_min", type=float, default=0.40, help="minimum percentage of length"
    )
    parser.add_argument(
        "--prob_max", type=float, default=0.50, help="maximum percentage of length"
    )

    # model sampling
    parser.add_argument(
        "--eval_max_length",
        type=int,
        default=512,
        help="maximum length of the generated text.",
    )
    parser.add_argument(
        "--eval_num_return_sequences",
        type=int,
        default=1,
        help="number of independent generations per story.",
    )
    parser.add_argument("--eval_num_beams", type=int, default=None)
    parser.add_argument(
        "--eval_temperature", type=float, default=None, help="sampling temperature."
    )
    parser.add_argument(
        "--eval_top_k",
        type=int,
        default=None,
        help="number of highest probability vocabulary tokens to keep for top-k-filtering.",
    )
    parser.add_argument(
        "--eval_top_p",
        type=float,
        default=None,
        help="cumulative probability for nucleus sampling.",
    )
    parser.add_argument("--eval_device", type=str, default="cuda", help="device to use")
    parser.add_argument(
        "--eval_batch_size", type=int, default=32, help="batch size for generation"
    )
    parser.add_argument("--eval_do_sample", type=bool, default=True, help="do sampling")
    parser.add_argument(
        "--eval_truncation", type=bool, default=True, help="truncate the input"
    )
    parser.add_argument(
        "--filetype",
        type=str,
        default="yaml",
        choices=["yaml", "json"],
        help="save outputs in yaml or json format",
    )

    config = parser.parse_args()

    # save for reproducibility
    config.hardcoded_OUTPUT_PATH = SAVE_PATH

    if config.models is None:
        raise ValueError("Please provide models to generate from")

    # models to include generations from
    models = config.models.split(",")
    models = [x.strip() for x in models]

    if config.seed is not None:
        seed_everything(config.seed)

    print("models:")
    print(models)
    outputs = get_stories(config, models)

    # save generation results
    save_dir = SAVE_PATH + f"/{config.folder_name}/"
    os.makedirs(save_dir, exist_ok=True)

    print("Sample output (before escape):")
    print(yaml.dump(outputs[0]))
    print(f"saving at: {save_dir}")

    if config.filetype == "yaml":
        with open(save_dir + "generated.yaml", "w") as f:
            sample_str = yaml.dump(
                outputs, default_flow_style=False, sort_keys=False, indent=4, width=100
            )
            # escape unicode characters
            sample_str = sample_str.replace("\\u201C", '"').replace("\\u201D", '"')
            f.write(sample_str)

    elif config.filetype == "json":
        with open(save_dir + "generated.json", "w") as f:
            sample_str = json.dumps(outputs, sort_keys=False, indent=4)
            # escape unicode characters
            sample_str = sample_str.replace("\\u201C", '"').replace("\\u201D", '"')
            f.write(sample_str)

    # write config file
    with open(save_dir + "config.yaml", "w") as f:
        config.config_file = config.config_file[0]._relative
        yaml.dump(config.__dict__, f)
