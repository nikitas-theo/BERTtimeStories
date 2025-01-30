"""
modified from LTG-BERT --- https://github.com/ltgoslo/ltg-bert
"""

import argparse
import os

from smart_open import open
from tokenizers import Tokenizer
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cached Dataset Creation")
    parser.add_argument(
        "--segments_path",
        type=str,
        default=None,
        help="Path to the segmented data file.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to the tokenizer JSON file.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Sequence length of each cached input sequence.",
    )

    args = parser.parse_args()

    SEQ_LEN = args.sequence_length - 2

    print("Loading tokenizer from file {}".format(args.tokenizer_path))
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # empty newlines separate documents
    documents = [[]]
    for line in tqdm(open(args.segments_path)):
        line = line.strip()
        if len(line) == 0:
            if len(documents[-1]) > 0:
                documents.append([])
            continue

        ids = tokenizer.encode(line, add_special_tokens=False).ids
        documents[-1].append(ids)

    dir_path = os.path.dirname(args.segments_path)
    dir_path = dir_path.replace("raw", "processed")
    os.makedirs(dir_path, exist_ok=True)

    [base, end] = os.path.splitext(args.segments_path)
    base = base.replace("raw", "processed")
    path_put = base + f"_cached_{args.sequence_length}" + end

    with open(path_put, "w") as f:
        for document in tqdm(documents):
            segment = []

            for i, sentence in enumerate(document):
                segment += sentence

                if len(segment) > SEQ_LEN:
                    segment = segment[:SEQ_LEN]
                    subwords = [tokenizer.id_to_token(token_id) for token_id in segment]

                    f.write(" ".join(subwords) + "\n")

                    segment = [s for s in sentence]

            if len(segment) > 0:
                segment = segment[:SEQ_LEN]
                subwords = [tokenizer.id_to_token(token_id) for token_id in segment]
                f.write(" ".join(subwords) + "\n")
