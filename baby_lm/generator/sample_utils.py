import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def sort_by_length(list_of_lists, ids=None):
    """
    Sort a list of lists by the length of the sublists.
    used for more efficient batch generation
    """
    np_array = np.array(list_of_lists, dtype=object)

    if ids is not None:
        sorted_ids = ids
    else:
        sorted_ids = np.argsort([len(sublist) for sublist in np_array])

    sorted_array = np_array[sorted_ids]
    sorted_list_of_lists = sorted_array.tolist()

    return sorted_list_of_lists, sorted_ids


def pad_sequences_left(sequences, padding_value=0):
    """
    Pads a list of 1D tensors to the length of the longest tensor, padding from the left.
    Args:
        sequences (list of torch.Tensor): List of 1D tensors of varying lengths.
        padding_value (int, optional): The value to use for padding. Default is 0.

    Returns:
        torch.Tensor: A single tensor with all sequences padded to the same length.
    """
    sequences = [torch.tensor(s) for s in sequences]
    # Find the maximum length
    max_len = max(seq.size(0) for seq in sequences)
    # Pad each tensor manually to the maximum length, padding from the left
    padded_sequences = [
        F.pad(seq, (max_len - seq.size(0), 0), value=padding_value) for seq in sequences
    ]
    # Stack the padded sequences into a single tensor
    padded_sequences = torch.stack(padded_sequences)

    return padded_sequences


def generate_sequences(prompts, model_load_path, generation_parameters_dict, HF=False):
    """
    Generate sequences using a pre-trained language model.

    Args:
        prompts (list of str): Input prompts for which to generate completions.
        model_load_path (str): Path to the pre-trained model.
        generation_parameters_dict (dict): Dictionary of generation parameters.
        HF (bool): If True, load the model and tokenizer from Hugging Face.

    Returns:
        list of dict: Generated sequences for each input prompt.
        Each dict contains:
            - prompt (str): The input prompt.
            - model_completion (list of str): List of model completions
            - generated_text d(list of str): List of final generated texts, i.e., prompt + model_completion
    """

    print(generation_parameters_dict)

    model = AutoModelForCausalLM.from_pretrained(model_load_path)
    tokenizer = AutoTokenizer.from_pretrained(model_load_path, padding_side="left")

    print(prompts[0])

    lens = [len(p.split()) for p in prompts]

    print(f"Using {len(prompts)} prompts")
    print("Max prompt length:", max(lens))
    print("Min prompt length:", min(lens))
    print("Mean prompt length:", sum(lens) / len(lens))

    t1 = time.time()

    # copy because we need them
    device = generation_parameters_dict["device"]
    truncation = generation_parameters_dict["truncation"]
    batch_size = generation_parameters_dict["batch_size"]

    # delete because model parameters do not include these
    del generation_parameters_dict["device"]
    del generation_parameters_dict["batch_size"]
    del generation_parameters_dict["truncation"]

    # prepare loop
    model = model.to(device)
    model = model.eval()
    pad_token_id = tokenizer.eos_token_id
    loop_size = len(prompts)

    # tokenize outside without padding
    inputs_model = tokenizer(
        prompts,
        max_length=generation_parameters_dict["max_length"],
        truncation=truncation,
    )

    # sort by length
    input_ids, sorted_ids = sort_by_length(inputs_model["input_ids"])
    attention_mask, _ = sort_by_length(inputs_model["attention_mask"], sorted_ids)

    inputs_model["input_ids"] = input_ids
    inputs_model["attention_mask"] = attention_mask

    prompts, _ = sort_by_length(prompts, sorted_ids)

    generations = []
    for i in tqdm(range(0, loop_size, batch_size)):
        # batchify
        input_ids_batch = inputs_model["input_ids"][i : i + batch_size]
        attention_mask_batch = inputs_model["attention_mask"][i : i + batch_size]

        # pad sequences inside loop
        input_ids_batch = pad_sequences_left(input_ids_batch, pad_token_id)

        attention_mask_batch = pad_sequences_left(attention_mask_batch, padding_value=0)

        with torch.no_grad():
            # move to device
            input_ids_batch = input_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)

            # generate for batch
            output_model = model.generate(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                **generation_parameters_dict,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=None,
            )

            # decode and save
            outputs_tok = tokenizer.batch_decode(output_model, skip_special_tokens=True)
            num_return_sequences = generation_parameters_dict["num_return_sequences"]

            # unpack generations
            for i in range(len(input_ids_batch)):
                generations.append(
                    [
                        {"generated_text": generation.strip()}
                        for generation in outputs_tok[
                            i * num_return_sequences : (i + 1) * num_return_sequences
                        ]
                    ]
                )

            assert len(generations[0]) == num_return_sequences
            output_model = output_model.cpu()
            input_ids_batch = input_ids_batch.cpu()
            attention_mask_batch = attention_mask_batch.cpu()

            del input_ids_batch, attention_mask_batch, output_model

    print("Time generation:", time.time() - t1)

    outputs = []
    print(len(generations))

    for p, gen in zip(prompts, generations):
        outputs.append({})

        outputs[-1]["prompt"] = p
        outputs[-1]["generated_text"] = []
        outputs[-1]["model_completion"] = []

        for g in gen:
            outputs[-1]["generated_text"].append(g["generated_text"])
            out_str = f'{g["generated_text"][len(p) :]}'
            outputs[-1]["model_completion"].append(out_str)

    # need to revert outputs to original order as later processing expects the order to stay the same
    outputs_original_order = [None] * len(sorted_ids)  # create a placeholder list
    for i, original_index in enumerate(sorted_ids):
        outputs_original_order[original_index] = outputs[i]

    return outputs_original_order
