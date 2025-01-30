import json
import os

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm


def main(args):
    """
    Calculates Self-BLEU of generations for GPT-Neo models trained on TinyStories.
    The average self-BLEU in the collection of `num_stories` prompts and `k` generations per prompt is calculated.
    The self-BLEU is calculated for k in [2, 3, 4, 5, 6, 7, 8, 9, ... ,`num_generated`]
    """
    model_str = args.model.replace("/", "_")
    folder_name = (
        f"diversity_{model_str}_stories{args.num_stories}_gen{args.num_generated}"
    )
    # sample stories from the model
    config_file = "./configs/sampling/dataset/generate_dataset_self_bleu.yaml"
    sample_from = f"./data/processed/tinystories/train_{args.millions}M/TinyStoriesV2-GPT4-train_{args.millions}M_proc.txt"
    cmd = [
        "python -m",
        "baby_lm.generator.sample",
        "--config_file",
        config_file,
        "--sample_from",
        sample_from,
        "--models",
        args.model,
        "--folder_name",
        folder_name,
        "--eval_num_return_sequences",
        str(args.num_generated),
        "--num_stories",
        args.num_stories,
        "--eval_batch_size",
        args.bs,
    ]
    cmd = [str(c) for c in cmd]
    cmd = " ".join(cmd)
    # os.system(cmd)

    with open(f"./data/raw/generated/{folder_name}/generated.json", "r") as f:
        generations = json.load(f)

    self_bleu_range = [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    ]
    self_bleu_range = [b for b in self_bleu_range if b <= args.num_generated]
    scores = {i: [] for i in self_bleu_range}

    # for each k value in the self_bleu_range
    for num_stories_bleu in self_bleu_range:
        print("Calculating BLEU for", num_stories_bleu)

        # for each prompt
        for gen in tqdm(generations):
            # get all the stories generated for this prompt
            stories_per_prompt = gen[args.model.replace("_HF", "")]

            blue_scores = []

            for story_id in range(num_stories_bleu):
                prediction = stories_per_prompt[story_id]
                references = [
                    stories_per_prompt[k]
                    for k in range(num_stories_bleu)
                    if k != story_id
                ]

                prediction = [prediction.split()]
                references = [[r.split() for r in references]]
                score = corpus_bleu(
                    hypotheses=prediction, list_of_references=references
                )
                blue_scores.append(score)
            # calculate the self_bleu score
            self_blue = np.mean(blue_scores)
            # add the self_bleu score to the list of scores for this k value
            scores[num_stories_bleu].append(self_blue)

    # check that we have the correct number of scores for each k value
    for i in self_bleu_range:
        assert len(scores[i]) == args.num_stories

    # get mean scores for all the k values
    mean_scores = {b: np.mean(scores[b]) for b in self_bleu_range}
    print("Sel-Bleu range, ", self_bleu_range)
    print("Mean scores, ", mean_scores)
    results = {"scores": scores, "mean_scores": mean_scores}

    os.makedirs(f"./outputs/evaluation/{folder_name}/", exist_ok=True)
    with open(f"./outputs/evaluation/{folder_name}/self_bleu_scores.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="""model name, can be a local model name in /outputs/models/<name>/ or a HF model 
        e.g., nikitastheo/GPT-Neo-50m_HF (_HF is added to the name to indicate HF model)""",
    )
    parser.add_argument(
        "--millions",
        type=str,
        default=None,
        help="millions of TinyStories data to use — must already be present in /data/processed/tinystories/",
    )
    parser.add_argument(
        "--num_stories",
        type=int,
        default=100,
        help="number of input stories to use — in the paper 100 was used",
    )
    parser.add_argument(
        "--num_generated",
        type=int,
        default=50,
        help="number of generations per input story (prompt) — in the paper 50 was used",
    )
    parser.add_argument("--bs", type=int, default=2, help="batch size for generation")

    args = parser.parse_args()
    main(args)
