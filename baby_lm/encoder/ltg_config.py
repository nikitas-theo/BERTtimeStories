"""
modified from LTG-BERT --- https://github.com/ltgoslo/ltg-bert
"""

import copy
import json
from io import open

import yaml


class LTGBERTConfig(object):
    """Configuration class to store the configuration of `LTGBERT` model."""

    def __init__(
        self,
        config_file=None,
        attention_probs_dropout_prob=None,
        hidden_dropout_prob=None,
        hidden_size=None,
        intermediate_size=None,
        max_position_embeddings=None,
        position_bucket_size=None,
        num_attention_heads=None,
        num_hidden_layers=None,
        vocab_size=None,
        layer_norm_eps=None,
    ):
        if config_file is not None:
            print("Loading LTGBERT config from", config_file)
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            for key, value in config.items():
                self.__dict__[key] = value

        else:
            print("Using agruments for LTGBERT config")
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.hidden_dropout_prob = hidden_dropout_prob
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.max_position_embeddings = max_position_embeddings
            self.position_bucket_size = position_bucket_size
            self.num_attention_heads = num_attention_heads
            self.num_hidden_layers = num_hidden_layers
            self.vocab_size = vocab_size
            self.layer_norm_eps = layer_norm_eps

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `LTGBERTConfig` from a Python dictionary of parameters."""
        config = cls(**json_object)
        return config

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
