"""
modified from LTG-BERT --- https://github.com/ltgoslo/ltg-bert
"""

import argparse
import os
import re

from smart_open import open

from baby_lm.process_data.normalize import clean_text


def process_bnc(f):
    prev_line = None
    for line in f:
        line = line.strip()

        if len(line) == 0:
            yield ""
            prev_line = None
            continue

        if line in [".", "!", "?"]:
            continue

        line = line[0].upper() + line[1:]
        line = clean_text(line)
        line = f'"{line}"'

        if prev_line is not None and prev_line == line:
            continue

        yield line
        prev_line = line


def process_childes(f):
    prev_line = None
    for line in f:
        line = line.strip()

        if len(line) == 0:
            yield ""
            continue

        line = line[0].upper() + line[1:]
        line = clean_text(line)
        line = f'"{line}"'

        if prev_line is not None and prev_line == line:
            continue

        yield line
        prev_line = line


def process_gutenberg(f):
    last_num_non_blank_lines = 0
    num_blank_lines = 0
    accumulated_line = []
    for line in f:
        line = " ".join(line.strip().split())
        line = clean_text(line, minimal=True)

        if len(line) == 0:
            if len(accumulated_line) > 0:
                yield " ".join(accumulated_line)
                last_num_non_blank_lines = len(accumulated_line)

            if num_blank_lines == 1 and last_num_non_blank_lines > 1:
                yield ""

            accumulated_line = []
            num_blank_lines += 1
            continue

        num_blank_lines = 0
        accumulated_line.append(line)


def proces_open_subtitles(f):
    prev_line = None
    for line in f:
        line = " ".join(line.strip().split())
        line = clean_text(line, minimal=True)

        if line.startswith("- "):
            line = line[2:]
        elif line.startswith("-"):
            line = line[1:]

        if len(line) == 0:
            if prev_line is None or len(prev_line) > 0:
                yield ""
            prev_line = line
            continue

        if (
            not line.endswith(":")
            and not line.startswith('"')
            and not line.endswith('"')
            and not (line.startswith("(") and line.endswith(")"))
            and not (line.startswith("[") and line.endswith("]"))
            and not (line.startswith("{") and line.endswith("}"))
        ):
            line = f'"{line}"'

        if prev_line is not None and prev_line == line:
            continue

        yield line
        prev_line = line


regex = re.compile(r"\[\d+\]")


def process_simple_wiki(f):
    prev_line = None
    for line in f:
        line = " ".join(line.strip().split())
        line = regex.sub("", line)
        line = clean_text(line, minimal=True)

        if len(line) == 0:
            if prev_line is not None and prev_line != "":
                yield prev_line
                yield ""
            prev_line = None
            continue

        if "is a commune. It is" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a commune found" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a city in" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a village in" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a municipality in" in line and len(line) < 128:
            prev_line = None
            continue

        if "is a town in" in line and len(line) < 128:
            prev_line = None
            continue

        line = line.replace("&gt;", ">")
        line = line.replace("&lt;", "<")
        line = line.replace("&amp;", "&")

        line = line.replace("&gt;", ">")
        line = line.replace("&lt;", "<")
        line = line.replace("&amp;", "&")

        if prev_line is not None:
            yield prev_line

        prev_line = line


def process_switchboard(f):
    prev_line = None
    for line in f:
        line = " ".join(line.strip().split())

        if line.startswith("- "):
            line = line[2:]
        elif line.startswith("-"):
            line = line[1:]

        line = clean_text(line, minimal=True)

        if len(line) == 0:
            yield ""
            continue

        line = f'"{line}"'

        if prev_line is not None and prev_line == line:
            continue

        yield line
        prev_line = line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="all",
        type=str,
        choices=[
            "all",
            "bnc_spoken",
            "childes",
            "gutenberg",
            "open_subtitles",
            "simple_wiki",
            "switchboard",
        ],
    )

    parser.add_argument(
        "--split",
        default="train_10M",
        type=str,
        choices=["train_10M", "train_100M", "dev", "test"],
    )

    args = parser.parse_args()

    if args.dataset == "all":
        process = [
            "bnc_spoken",
            "childes",
            "gutenberg",
            "open_subtitles",
            "simple_wiki",
            "switchboard",
        ]
    else:
        process = [args.dataset]

    if args.split in ["train_10M", "train_100M"]:
        suffix = "train"
    elif args.split in ["test"]:
        suffix = "test"
    else:
        suffix = "dev"

    for dataset in process:
        if dataset == "bnc_spoken":
            process = process_bnc
        elif dataset == "childes":
            process = process_childes
        elif dataset == "gutenberg":
            process = process_gutenberg
        elif dataset == "open_subtitles":
            process = proces_open_subtitles
        elif dataset == "simple_wiki":
            process = process_simple_wiki
        elif dataset == "switchboard":
            process = process_switchboard
        else:
            raise ValueError(f"Unknown dataset {dataset}")

        path = f"./data/raw/babylm/{args.split}/{dataset}.{suffix}"

        print(f"Processing {dataset} with path {path}")

        path_put = path.replace("/raw/", "/processed/")

        os.makedirs(os.path.dirname(path_put), exist_ok=True)
        with open(path, "r") as f:
            with open(path_put, "w") as g:
                for line in process(f):
                    g.write(f"{line}\n")
