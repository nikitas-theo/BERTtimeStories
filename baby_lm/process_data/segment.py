"""
modified from LTG-BERT --- https://github.com/ltgoslo/ltg-bert
"""

import argparse
import os

import nltk
import tqdm


def segment(load_path, save_path, special_token):
    print("loading from: ", load_path)
    print("saving at:", save_path)
    with open(save_path, "w") as f:
        for line in tqdm.tqdm(open(load_path)):
            line = line.strip()

            # keep newlines
            if len(line) == 0:
                f.write("\n")
            else:
                sentences = nltk.sent_tokenize(line)
                sentences = "\n".join(sentences)
                f.write(f"{sentences}{special_token}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="concatenated dataset file, all.txt")
    parser.add_argument("--model_type", default=None, choices=["bert", "gpt"])

    args = parser.parse_args()

    if args.model_type == "bert":
        special_token = "[PAR]"
    else:
        special_token = ""

    output_file = args.filepath.replace("raw", "processed")
    dir_path = os.path.dirname(output_file)
    os.makedirs(dir_path, exist_ok=True)

    [base, end] = os.path.splitext(output_file)
    output_file = base + "_segmented" + end
    segment(args.filepath, output_file, special_token=special_token)
