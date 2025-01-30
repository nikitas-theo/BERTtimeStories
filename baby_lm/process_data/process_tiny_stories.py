import argparse
import os

from tqdm import tqdm

from baby_lm.process_data.normalize import clean_text


def process_tinystories(f):
    for line in f:
        line = line.strip()

        if len(line) == 0:
            continue

        line = clean_text(line)

        yield line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)

    args = parser.parse_args()
    with open(args.path, "r") as f:
        path_put = args.path.replace("raw", "processed")
        [base, end] = os.path.splitext(path_put)
        folder = os.path.dirname(path_put)
        os.makedirs(folder, exist_ok=True)
        path_put = base + "_proc" + end

        with open(path_put, "w") as g:
            for i, line in enumerate(tqdm(process_tinystories(f))):
                g.write(f"{line}\n")
