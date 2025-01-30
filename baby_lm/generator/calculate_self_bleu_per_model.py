import json
import os

import numpy as np
from nltk.translate.bleu_score import corpus_bleu


def calculate_self_bleu(results, model):
    n_stories = len(results)
    generations = [results[i][model][0] for i in range(n_stories)]
    scores = []

    for i, story in enumerate(generations):
        prediction = story
        references = [generations[k] for k in range(n_stories) if k != i]
        score = corpus_bleu(
            hypotheses=[prediction.split()],
            list_of_references=[[r.split() for r in references]],
        )
        print(f"BLEU score for story={i}th prediction: {score}")
        scores.append(score)

    self_bleu = np.mean(scores)
    self_bleu = float(self_bleu)

    print(self_bleu)

    return self_bleu


def main():
    models_HF = [
        "roneneldan/TinyStories-33M_HF",
        "nikitastheo/GPT-Neo-440m_HF",  # corresponds to 500M (rounded)
        "nikitastheo/GPT-Neo-100m_HF",
        "nikitastheo/GPT-Neo-75m_HF",
        "nikitastheo/GPT-Neo-50m_HF",
        "nikitastheo/GPT-Neo-25m_HF",
        "nikitastheo/GPT-Neo-10m_HF",
        "nikitastheo/GPT-Neo-5m_HF",
    ]

    # models need to be present with the format: outputs/models/gpt_neo_50M/checkpoint/
    models_local = [
        # 33M refers to the model parameters
        "roneneldan/TinyStories-33M_HF",
        "gpt_neo_500M",
        "gpt_neo_100M",
        "gpt_neo_75M",
        "gpt_neo_50M",
        "gpt_neo_25M",
        "gpt_neo_10M",
        "gpt_neo_5M",
    ]
    # change between local or HF
    models = models_HF
    models = [model.strip() for model in models]
    # files to sample stories and creates prompt for generation
    # for each model, we choose stories from its training data
    sample_files = [
        f"./data/processed/tinystories/train_{millions}/TinyStoriesV2-GPT4-train_{millions}_proc.txt"
        for millions in ["500M", "100M", "75M", "50M", "25M", "10M", "5M"]
    ]
    # original TinyStories model (roneneldan/TinyStories-33M_HF) is trained on the complete original TinyStories dataset
    sample_files = ["./data/raw/tinystories/TinyStories-train.txt"] + sample_files

    config_file = "./configs/sampling/evaluation/paper_calculate_bleu.yaml"
    self_bleu_scores = {}

    # sample stories for our models
    for sample_file, model in zip(sample_files, models):
        folder_name = f"generations_bleu_{model}"
        folder_name = folder_name.replace("/", "_")
        cmd = f"""
            python -m baby_lm.generator.sample \\
            --config_file {config_file} \\
            --model {model} \\
            --folder_name {folder_name} \\
            --sample_from {sample_file}
        """
        print(cmd)
        os.system(cmd)

        with open(f"./data/raw/generated/{folder_name}/generated.json", "r") as f:
            results = json.load(f)
        model = model.replace("_HF", "")

        # calculate self bleu
        self_bleu = calculate_self_bleu(results, model)
        print(f"Self-BLEU for {model}: {self_bleu}")
        self_bleu_scores[model] = self_bleu

    # save results for all models
    save_path = "./outputs/evaluation/self_bleu_scores_all_models/"
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving results at {save_path}scores.json")
    with open(save_path + "scores.json", "w") as f:
        json.dump(self_bleu_scores, f, indent=4)


if __name__ == "__main__":
    main()
