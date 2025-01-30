import argparse
import gzip
import json
import os

import numpy as np

TEXT_TASKS = {
    "glue": [
        "cola",
        "sst2",
        "mrpc",
        "qqp",
        "mnli",
        "mnli-mm",
        "qnli",
        "rte",
        "boolq",
        "multirc",
        "wsc",
    ],
    # Lots of BLiMP tasks – use verifier function below to see if you've included everything.
    "blimp": [
        taskname.split(".jsonl")[0]
        for taskname in os.listdir("evaluation_data/blimp_filtered/")
    ],
    "blimp_supplement": [
        "hypernym",
        "qa_congruence_easy",
        "qa_congruence_tricky",
        "subject_aux_inversion",
        "turn_taking",
    ],
    "ewok": [
        "agent-properties",
        "material-dynamics",
        "material-properties",
        "physical-dynamics",
        "physical-interactions",
        "physical-relations",
        "quantitative-properties",
        "social-interactions",
        "social-properties",
        "social-relations",
        "spatial-relations",
    ],
}

VISION_TASKS = {
    "vqa": ["vqa"],
    "winoground": ["winoground"],
    "devbench": ["lex-viz_vocab", "gram-trog", "sem-things"],
}


def zeroshot_harness_parser(predictions_file):
    predictions = []
    json_obj = json.load(predictions_file)
    for document in json_obj:
        logprobs = [resp[0][0] for resp in document["resps"]]
        top_logprob_idx = np.argmax(logprobs)
        top_response = document["arguments"][top_logprob_idx][1]
        predictions.append(top_response)
    return predictions


def glue_parser(predictions_file):
    next(predictions_file)  # skip header
    predictions = []
    for line in predictions_file:
        index, prediction = line.strip().split("\t")
        predictions.append(int(prediction))
    return predictions


def devbench_parser(predictions_filename):
    prediction_matrix = np.load(predictions_filename)
    prediction_matrix = prediction_matrix.tolist()
    return prediction_matrix


def make_task_dict(task_name, subtask_name, preds_path):
    def _add_to_dict(index, prediction, task_dict):
        example_id = f"{subtask_name}_{index}"
        if type(prediction) == str:
            prediction = prediction.replace("\\n", "\n")
        task_dict["predictions"].append({"id": example_id, "pred": prediction})

    if not os.path.exists(preds_path):
        raise FileNotFoundError(
            f'Error: no predictions found for "{subtask_name}" (in {task_name}).'
        )

    task_dict = {"predictions": []}
    with open(preds_path, "r") as predictions_file:
        if task_name in ("blimp", "blimp_supplement", "ewok", "vqa", "winoground"):
            predictions = zeroshot_harness_parser(predictions_file)
        elif task_name == "glue":
            predictions = glue_parser(predictions_file)
        else:
            prediction_matrix = devbench_parser(preds_path)
            task_dict["predictions"] = prediction_matrix
            return task_dict

        for idx, prediction in enumerate(predictions):
            _add_to_dict(idx, prediction, task_dict)

    return task_dict


def verify_dict(task_dicts, includes_vision_tasks):
    # Verify all required tasks are present
    for task in TEXT_TASKS:
        assert task in task_dicts, f"Error: {task} not present."
        for subtask in TEXT_TASKS[task]:
            assert (
                subtask in task_dicts[task]
            ), f"Error: {subtask} not present under {task}."
    if includes_vision_tasks:
        for task in VISION_TASKS:
            assert task in task_dicts, f"Error: {task} not present."
            for subtask in VISION_TASKS[task]:
                assert (
                    subtask in task_dicts[task]
                ), f"Error: {subtask} not present under {task}."

    # Make sure all examples have predictions, and that predictions are the correct type
    for task in task_dicts:
        for subtask in task_dicts[task]:
            if task == "devbench":
                a = np.array(task_dicts[task][subtask]["predictions"])
                if subtask == "sem-things":
                    required_shape = (1854, 1854)
                elif subtask == "gram-trog":
                    required_shape = (76, 4, 1)
                elif subtask == "lex-viz_vocab":
                    required_shape = (119, 4, 1)
                if a.shape[0] != required_shape[0] or a.shape[1] != required_shape[1]:
                    raise Exception(
                        f"Error: Wrong shape for results for `{subtask}` in `{task}`."
                    )
                assert str(a.dtype).startswith(
                    "float"
                ), f"Error: Results for `{subtask}` ({task}) \
                        should be floats but aren't."
                continue

            task_folder = (
                f"{task}_filtered"
                if task != "blimp_supplement"
                else "supplement_filtered"
            )
            if task == "glue":
                filename = f"evaluation_data/{task_folder}/{subtask}.valid.jsonl"
                assert (
                    type(task_dicts[task][subtask]["predictions"][0]["pred"]) == int
                ), f"Error: \
                                Results for `{subtask}` ({task}) should be integers but aren't."
            else:
                if subtask == "vqa":
                    filename = (
                        f"evaluation_data/{task_folder}/vqa_distractors_info.json"
                    )
                    with open(filename, "r") as subtask_data:
                        json_data = json.load(subtask_data)
                        num_examples = len(json_data)
                        if (
                            len(task_dicts[task][subtask]["predictions"])
                            != num_examples
                        ):
                            raise Exception(
                                f"Error: Examples missing for `{subtask}` in `{task}`."
                            )
                    assert (
                        type(task_dicts[task][subtask]["predictions"][0]["pred"]) == str
                    ), f"Error: \
                                Results for `{subtask}` ({task}) should be strings but aren't."
                    continue
                else:
                    filename = f"evaluation_data/{task_folder}/{subtask}.jsonl"
                    assert (
                        type(task_dicts[task][subtask]["predictions"][0]["pred"]) == str
                    ), f"Error: \
                                Results for `{subtask}` ({task}) should be strings but aren't."
            with open(filename, "r") as subtask_data:
                num_examples = len(subtask_data.readlines())
                if len(task_dicts[task][subtask]["predictions"]) != num_examples:
                    raise Exception(
                        f"Error: Examples missing for `{subtask}` in `{task}`."
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        type=str,
        help="The name of your model. Should correspond to a "
        "directory name in `results/`.",
    )
    parser.add_argument(
        "--include_vision_tasks",
        "-v",
        action="store_true",
        help="Whether to include vision tasks in the final output.",
    )
    parser.add_argument(
        "--glue_lora",
        "-l",
        action="store_true",
        help="If True, look for GLUE results in `results/lora` instead of default `results/finetune`.",
    )
    args = parser.parse_args()

    model_basename = args.model_name.split("/")[-1]
    task_dicts = {"glue": {}, "blimp": {}, "blimp_supplement": {}, "ewok": {}}

    # Build task predictions dictionaries
    for task in TEXT_TASKS["glue"]:
        results_dir = "finetune"
        if args.glue_lora:
            results_dir = "lora"
        preds_path = f"results/{results_dir}/{model_basename}/{task}/predictions.txt"
        task_dicts["glue"][task] = make_task_dict("glue", task, preds_path)
    for task in TEXT_TASKS["blimp"]:
        preds_path = (
            f"results/blimp/{model_basename}/blimp_{task}_filtered_results.jsonl"
        )
        task_dicts["blimp"][task] = make_task_dict("blimp", task, preds_path)
    for task in TEXT_TASKS["blimp_supplement"]:
        preds_path = (
            f"results/blimp/{model_basename}/blimp_supplement_{task}_results.jsonl"
        )
        task_dicts["blimp_supplement"][task] = make_task_dict(
            "blimp_supplement", task, preds_path
        )
    for task in TEXT_TASKS["ewok"]:
        preds_path = f"results/ewok/{model_basename}/ewok_{task}_filtered_results.jsonl"
        task_dicts["ewok"][task] = make_task_dict("ewok", task, preds_path)

    if args.include_vision_tasks:
        task_dicts["vqa"] = {}
        task_dicts["vqa"]["vqa"] = make_task_dict(
            "vqa",
            "vqa",
            f"results/vqa_filtered/{model_basename}/vqa_val_filtered_results.jsonl",
        )
        task_dicts["winoground"] = {}
        task_dicts["winoground"]["winoground"] = make_task_dict(
            "winoground",
            "winoground",
            f"results/winoground_filtered/{model_basename}/winoground_results.jsonl",
        )
        task_dicts["devbench"] = {}
        for task in VISION_TASKS["devbench"]:
            if task == "sem-things":
                preds_path = f"evaluation_data/devbench/evals/sem-things/{model_basename}_pairwise_sims.npy"
            else:
                preds_path = (
                    f"evaluation_data/devbench/evals/{task}/{model_basename}.npy"
                )
            task_dicts["devbench"][task] = make_task_dict("devbench", task, preds_path)

    # Save predictions
    preds_name = "withvision" if args.include_vision_tasks else "textonly"
    with gzip.open(
        f"{model_basename}_{preds_name}_predictions.json.gz", "wt"
    ) as predictions_out:
        json.dump(task_dicts, predictions_out)

    # Make sure dictionary includes everything and is formatted correctly
    verify_dict(task_dicts, args.include_vision_tasks)
