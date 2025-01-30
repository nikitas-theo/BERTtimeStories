import argparse
import os

import tokenizers
import torch
from tokenizers import Tokenizer
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from baby_lm.encoder.ltg_config import LTGBERTConfig
from baby_lm.encoder.ltg_model import LTGBERT


def prepare_GPT(checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.add_special_tokens({"bos_token": "<|endoftext|>"})
    tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    tokenizer.add_special_tokens({"unk_token": "[UNK]"})

    tokenizer.save_pretrained(checkpoint_path)


def prepare_LTG_BERT(checkpoint_path):
    def load_LTG_BERT_HF(model_config_dict):
        model_config_hf = AutoConfig.from_pretrained(
            "babylm/ltgbert-100m-2024", trust_remote_code=True
        )
        # update model config values with custom config
        for key, value in model_config_dict.items():
            setattr(model_config_hf, key, value)

        model_hf = AutoModelForMaskedLM.from_config(
            model_config_hf, trust_remote_code=True
        )
        return model_hf

    def load_LTG_BERT(checkpoint_path):
        checkpoint = torch.load(checkpoint_path + "checkpoint.pth", map_location="cpu")
        print(checkpoint.keys())
        model_config_dict = checkpoint["model_config"]

        model_config = LTGBERTConfig.from_dict(model_config_dict)
        model = LTGBERT(model_config)
        model_state_dict = checkpoint["model"]

        # remove DDP wrapping
        new_state_dict = {}
        for key in model_state_dict:
            if key.startswith("module."):
                new_key = key[len("module.") :]  # remove 'module.' from the key
                new_state_dict[new_key] = model_state_dict[key]
            else:
                new_state_dict[key] = model_state_dict[key]

        model_state_dict = new_state_dict
        model.load_state_dict(model_state_dict)

        tokenizer = Tokenizer.from_file(checkpoint_path + "tokenizer.json")

        return model, tokenizer, model_config_dict

    model_ltg, tokenizer_ltg, model_config_dict = load_LTG_BERT(checkpoint_path)
    model_hf = load_LTG_BERT_HF(model_config_dict)

    ##################################################
    # Get the keys of the state_dicts
    keys1 = set(model_hf.state_dict().keys())
    keys2 = set(model_ltg.state_dict().keys())

    # Compare the keys
    if keys1 == keys2:
        print("The models have the same state_dict keys.")
    else:
        print("The models do not have the same state_dict keys.")
        print("Keys in model1 but not in model2:", keys1 - keys2)
        print("Keys in model2 but not in model1:", keys2 - keys1)
    ##################################################

    model_hf.load_state_dict(model_ltg.state_dict())

    # save model in HuggingFace format
    print("saving model")
    model_hf.save_pretrained(checkpoint_path, safe_serialization=False)

    ##################################################
    # update tokenizer
    tokenizer_ltg.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $0 [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer_ltg.token_to_id("[CLS]")),
            ("[SEP]", tokenizer_ltg.token_to_id("[SEP]")),
        ],
    )

    # intermediate temp file
    tokenizer_ltg.save(checkpoint_path + "tokenizer_raw.json")
    tokenizer_ltg = PreTrainedTokenizerFast(
        tokenizer_file=checkpoint_path + "tokenizer_raw.json"
    )
    os.system(f"rm {checkpoint_path + 'tokenizer_raw.json'}")

    # moving special tokens here
    tokenizer_ltg.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer_ltg.add_special_tokens({"unk_token": "[UNK]"})
    tokenizer_ltg.add_special_tokens({"cls_token": "[CLS]"})
    tokenizer_ltg.add_special_tokens({"sep_token": "[SEP]"})
    tokenizer_ltg.add_special_tokens({"mask_token": "[MASK]"})
    ##################################################

    print("saving tokenizer")
    tokenizer_ltg.save_pretrained(checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="e.g., outputs/models/GPT-Neo-5m/checkpoint/",
    )
    parser.add_argument(
        "--model_type", type=str, help="gpt or ltg-bert", choices=["gpt", "ltg-bert"]
    )

    args = parser.parse_args()

    model_type = args.model_type
    print(model_type)
    if model_type == "gpt":
        prepare_GPT(args.checkpoint_path)
    elif model_type == "ltg-bert":
        prepare_LTG_BERT(args.checkpoint_path)
    else:
        raise ValueError(f"Model type {model_type} not supported")

    model_name = args.checkpoint_path.split("/")[-3]
    print(model_name)
    print("Success! Model is ready for evaluation")
