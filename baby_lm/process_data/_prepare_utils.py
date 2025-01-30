import itertools
import os
import random
import subprocess

import numpy as np
from tqdm import tqdm


def seed_everything(seed_value):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)


# function to execute shell commands
def run_command(command):
    print(command)
    process = subprocess.Popen(command, shell=True)
    process.communicate()


# function to execute shell commands and return the output
def run_command_capture_output(command):
    print(command)
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(f"Command failed with error: {stderr.decode('utf-8')}")
    return stdout.decode("utf-8").strip()


def sample_dataset(
    input_file, output_file, num_words, special=None, shuffle_documents=False
):
    # clear the output file
    with open(output_file, "w") as f_out:
        f_out.write("")

    if shuffle_documents:
        # load documents
        documents = [[]]
        for line in tqdm(open(input_file, "r")):
            line = line.strip()
            # found a new document
            if len(line) == 0:
                if len(documents[-1]) > 0:
                    documents.append([])
            documents[-1].append(line)

        # shuffle at the document level to ensure a fair sampling
        random.shuffle(documents)

        lines_flat = list(itertools.chain(*documents))
    else:
        lines_flat = open(input_file, "r").readlines()

    # exclude special tokens when sampling for a specific number of words
    # i.e., if you want 100K words, then <|endoftext|> should not be counted
    num_words_counted = 0

    for line in lines_flat:
        line = line.strip()
        if special is not None:
            len_line = len(line.replace(special, "").split())
        else:
            len_line = len(line.split())

        num_words_counted += len_line
        if num_words_counted > num_words:
            print(
                f"Reached the number of words {num_words_counted} > {num_words}. Last line had {len_line} words."
            )
            break
        else:
            with open(output_file, "a") as f_out:
                f_out.write(line + "\n")

    number_words_new = run_command_capture_output(
        f"wc -w {output_file} | cut -d ' ' -f 1"
    )
    number_words_new = int(number_words_new)

    if special is not None:
        assert special == "<|endoftext|>", "Unknown special token."
        number_of_occurences = run_command_capture_output(
            f"grep -o '{special}' {output_file} | wc -l"
        )
        number_of_occurences = int(number_of_occurences)
        print(f"Number of occurences of '{special}': {number_of_occurences}")
        print(f"Counts special token: {special} = {number_of_occurences} times.")
        assert (
            number_words_new <= num_words + number_of_occurences
        ), f"Sampled more words than expected: {number_words_new} > {num_words}"

    print(f"Sampled {num_words} words with special {special}.")
    print(f"File includes {number_words_new} words.")
