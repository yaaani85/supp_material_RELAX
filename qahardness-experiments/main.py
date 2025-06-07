import os
import os.path as osp
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Define the base directories
methods_printnames = [
    "RELAX",
    "ULTRA",
    "ConE",
    "QTO"
]
methods = [m.lower() for m in methods_printnames]

# Define query structures of interest
query_structures = ["1p", "2p", "3p", "2i", "3i", "ip", "pi"]

# Dictionary to store mappings
query_ranks = {m: {} for m in methods}

for method in methods:
    method_path = osp.join('answers', method)
    for dataset in os.listdir(method_path):
        dataset_path = os.path.join(method_path, dataset)
        query_ranks[method][dataset] = {}

        for query in query_structures:
            query_folder = os.path.join(dataset_path, query)
            query_file = os.path.join(query_folder, "query_answer_ranks.pkl")
            if os.path.exists(query_file):
                query_ranks[method][dataset][query] = query_file

# Print the mappings
for method, mapping in query_ranks.items():
    print(f"{method} mappings:")
    for dataset, structure_to_file in mapping.items():
        print(f"\t{dataset}")
        for structure, file in structure_to_file.items():
            with open(file, "rb") as f:
                data_size = len(p.load(f))
            print(f"\t\t{structure}: {file}, {data_size:,} queries")
        print()


def compute_jaccard_similarity(results_A, results_B, k):
    """
    Computes the average Jaccard similarity between the top-k results of two systems for a given query structure.

    Parameters:
    - results_A (dict): A dictionary mapping query structures to the rankings of hard answers for system A
    - results_B (dict): Same, for system B
    - k (int): The ranking threshold to consider.

    Returns:
    - float: The average Jaccard similarity over all query IDs.
    """
    jaccard_similarities = []
    for q in results_A:
        rankings_a = results_A[q]
        rankings_b = results_B[q]

        # Both keys (hard answers) should be identical
        assert rankings_a.keys() == rankings_b.keys()

        top_k_a, top_k_b = [set([e for e, r in rankings.items() if r <= k]) for rankings in (rankings_a, rankings_b)]

        len_intersection = len(top_k_a & top_k_b)
        len_union = len(top_k_a | top_k_b)

        if len_union > 0:
            similarity = len_intersection / len_union
            jaccard_similarities.append(similarity)

    return np.mean(jaccard_similarities)


# --- Begin new plotting section ---
num_methods = len(methods)
fig, axes = plt.subplots(1, num_methods-1, figsize=(10, 3))

k_values = range(1, 100, 5)
dataset = "FB15k237+H"

method_A = methods[0]
for j, method_B in enumerate(methods[1:], start=1):
    ax = axes[j-1]
    all_sim_at_k = []
    for s in query_structures:
        file_A = query_ranks[method_A][dataset][s]
        file_B = query_ranks[method_B][dataset][s]
        with open(file_A, 'rb') as f:
            results_A = p.load(f)
        with open(file_B, 'rb') as f:
            results_B = p.load(f)
        sim_at_k = [compute_jaccard_similarity(results_A, results_B, k) for k in k_values]
        all_sim_at_k.append(sim_at_k)
        ax.plot(k_values, sim_at_k, label=s)

    mean_sim_at_k = np.mean(all_sim_at_k, axis=0)
    ax.plot(k_values, mean_sim_at_k, label='avg', color='black', linestyle='--', linewidth=3, alpha=0.5)
    ax.set_title(f"{methods_printnames[j] if methods_printnames[j] != 'ULTRA' else 'ULTRAQ'}")
    ax.set_xlabel('k')
    ax.set_ylim(0, 1)
    ax.grid(True)
    if j == num_methods-1:
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1))


plt.tight_layout()
plt.savefig("jaccard_similarity.pdf")
