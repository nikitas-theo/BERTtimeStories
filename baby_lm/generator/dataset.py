import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class GPTDataset(Dataset):
    def __init__(
        self,
        n_gpus: int,
        offset: int,
        seq_length,
        file,
        tokenizer,
        vocab_size,
    ):
        print(f"Loading dataset from {file}.")
        self.seq_length = seq_length
        self.file = file
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

        self.pad_index = 0

        self.segments = []

        # get pre-tokenized segments, and map to ids
        # <|endoftext|> should be present
        for i, segment in tqdm(enumerate(open(file, "r"))):
            if i % n_gpus != offset:
                # if n_gpus = 1, this should not be reached
                if n_gpus == 1:
                    raise Exception("One GPU, nothing should be skipped!")
                continue

            segment = segment.strip().split(" ")
            segment = [self.tokenizer.token_to_id(token) for token in segment]

            id_min = np.min(segment)
            id_max = np.max(segment)

            if id_min < 0 or id_max > self.vocab_size - 1:
                print("Invalid token id")
                import pdb

                pdb.set_trace()

            self.segments.append(segment)

    def __getitem__(self, index):
        tokens = self.segments[index]
        padding_length = (self.seq_length) - len(tokens)
        segment = tokens + [self.pad_index] * padding_length
        segment = torch.LongTensor(segment)

        outputs = segment

        # ignore pad
        attention_mask = torch.ones(self.seq_length)
        attention_mask[len(tokens) :] = 0

        if len(segment) > self.seq_length:
            print("Wrong segment length")
            import pdb

            pdb.set_trace()

        return segment, attention_mask, outputs

    def __len__(self):
        return len(self.segments)
