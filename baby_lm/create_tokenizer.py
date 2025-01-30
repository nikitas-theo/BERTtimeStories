"""
modified from LTG-BERT --- https://github.com/ltgoslo/ltg-bert
"""

import argparse
import os

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer


def init_tokenizer(vocab_size, min_frequency, model_type):
    if model_type == "gpt":
        special_tokens = ["[UNK]", "<|endoftext|>"]
    elif model_type == "ltg-bert":
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[PAR]"]

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        continuing_subword_prefix="",  # since we already use Ġ
    )

    tokenizer.normalizer = normalizers.NFD()

    # Pretokenize with ByteLevel, and Digits (just separates "123" -> "1 2 3")
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.ByteLevel(
                add_prefix_space=True,  # treat "_hello" same as "hello"
                use_regex=True,  # use GPT-2 regex for spliting on space
            ),
            pre_tokenizers.Digits(individual_digits=True),
        ]
    )

    tokenizer.decoder = decoders.ByteLevel()

    # trims offsets, used for indexing on the output
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    return tokenizer, trainer


def train_tokenizer(args):
    tokenizer, trainer = init_tokenizer(
        args.vocab_size, args.min_frequency, args.model_type
    )

    def iterator(file_path):
        for line in open(file_path, "r"):
            line = line.strip()
            if len(line) == 0:
                continue
            yield line

    print("Training the Tokenizer..", flush=True)
    tokenizer.train_from_iterator(iterator(args.input_path), trainer)

    print("Saving the tokenizer..", flush=True)

    # make sure path exists
    directory = os.path.dirname(args.vocab_path)
    os.makedirs(directory, exist_ok=True)

    tokenizer.save(args.vocab_path)

    print("TEST")
    print("Trying to load the tokenizer...")

    tokenizer = Tokenizer.from_file(args.vocab_path)

    print("Success!")

    # original test examples
    texts = [
        """One of the most impressive long term hobby projects is Robert's Rocket Project. He started building a 100 lbf liquid engine in 2001, fired a regeneratively cooled version in 2007, started building a regen 250 lbf in 2008.""",
        """what are examples of interfaces that allow you to manage sets of queries (SQL, splunk, lucene/elastic, xpath, whatever other language)?""",
        """### Increasingly seeing a big schism between what I think my research is & what others think it is. I don't do qualitative work and I'm not trained in anthro or theories of race or gender. I can't supervise students with these interests! I'm a sociophonetician who works on prosody!""",
        """The Northern Lights season is here... Taking these pictures is an art itself and requires preparation, so The Local spoke to an expert to find out how to take awe-inspiring snaps of the Northern Lights.""",
        """Some people have SOTA facial recognition abilities: "At the very upper end of the performance scale, a cohort of just 1-2% of the population are 'super-recognisers'-people who can memorise and recall unfamiliar faces, even after the briefest glimpse.\"""",
    ]

    def test(tokenizer, text):
        tokens = tokenizer.encode(text).tokens
        return " ".join(tokens)

    for text in texts:
        print(f"INPUT:  {text}\nTOKENS: {test(tokenizer, text)}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="Specify the input filename for the training data (.txt)",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        help="Specify the output filename for the tokenizer file (.json)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=2**14,
        help="Number of subwords in the trained tokenizer",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=10,
        help="Minimal number of occurences of every candidate subword",
    )
    parser.add_argument(
        "--model_type", type=str, default=None, choices=["gpt", "ltg-bert"]
    )

    args = parser.parse_args()
    train_tokenizer(args)
