import os
import pickle as p
from tqdm import tqdm
import torch
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model", choices=["relax", "qto", "cone", "ultra"], required=True)
parser.add_argument("--input", choices=["scores", "unfiltered_ranks"], default="scores", type=str)
args = parser.parse_args()

query_structures = ["1p", "2p", "3p", "2i", "3i", "ip", "pi"]

model_name = args.model

cr_path = os.path.join("answers", model_name)
for dataset in os.listdir(cr_path):
    dataset_path = os.path.join(cr_path, dataset)
    print(f"Loading {dataset}...")

    with open(os.path.join('data', dataset, 'test-easy-answers.pkl'), 'rb') as f:
        easy_answers = p.load(f)

    with open(os.path.join('data', dataset, 'test-hard-answers.pkl'), 'rb') as f:
        hard_answers = p.load(f)

    for structure in query_structures:
        query_file = os.path.join(dataset_path, structure, f"{model_name}_scores.pkl")

        with open(query_file, "rb") as f:
            query_scores = p.load(f)

        query_answer_ranks = dict()
        for query, scores in tqdm(query_scores.items(), desc=f"Processing {structure}", mininterval=1.0):
            if len(scores) == 1:
                scores = scores[0]
            easy = list(easy_answers[query])
            hard = list(hard_answers[query])
            num_easy = len(easy)
            num_hard = len(hard)

            if args.input == "scores":
                sorted_ids = torch.argsort(torch.tensor(scores), descending=True)
                rankings = torch.argsort(sorted_ids)
            elif args.input == "unfiltered_ranks":
                rankings = torch.tensor(scores, dtype=torch.long)

            # Note order: first easy answers, then hard
            answer_ranks = rankings[easy + hard]
            sorted_ranks, indices = torch.sort(answer_ranks)

            answer_list = torch.arange(num_easy + num_hard, dtype=torch.long)
            filtered_ranks = sorted_ranks - answer_list + 1
            # Recover original order
            filtered_ranks = filtered_ranks[indices.argsort()].tolist()

            # Based on the order above: [easy | hard], we get [hard] only
            query_answer_ranks[query] = {hard[i]: rank for i, rank in enumerate(filtered_ranks[num_easy:])}

        with open(os.path.join(dataset_path, structure, "query_answer_ranks.pkl"), "wb") as f:
            p.dump(query_answer_ranks, f)
