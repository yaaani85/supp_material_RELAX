import argparse
import pickle as p
import numpy as np
import torch
from scipy.stats import ttest_rel, sem, t
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from pprint import pprint


def mean_ci(data):
    m = np.mean(data)
    se_ = sem(data)
    ci = se_ * t.ppf((1 + 0.95) / 2., len(data)-1)
    return m, m-ci, m+ci


def significance_marker(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

def main(args):
    # Load pickled MRR dictionaries
    with open(args.mrr1, 'rb') as f:
        per_structure_mrr_1 = p.load(f)
    with open(args.mrr2, 'rb') as f:
        per_structure_mrr_2 = p.load(f)
    with open(args.mrr_combined, 'rb') as f:
        per_structure_mrr_combined = p.load(f)

    query_structures = args.query_structures.split(',')
    model_name_1 = args.model_1
    model_name_2 = args.model_2

    # Perform t-tests and Holm-Bonferroni correction
    p_values = []
    test_labels = []

    for structure in query_structures:
        mrr_1 = torch.tensor(per_structure_mrr_1[structure])
        mrr_2 = torch.tensor(per_structure_mrr_2[structure])
        mrr_combined = torch.tensor(per_structure_mrr_combined[structure])

        # Only run t-test if lengths match (should always be true, but safe check)
        if len(mrr_1) == len(mrr_combined):
            _, p1 = ttest_rel(mrr_1, mrr_combined)
            p_values.append(p1)
            test_labels.append(f"{structure}: model_1 vs combined")

        if len(mrr_2) == len(mrr_combined):
            _, p2 = ttest_rel(mrr_2, mrr_combined)
            p_values.append(p2)
            test_labels.append(f"{structure}: model_2 vs combined")

    # Holm–Bonferroni correction
    reject, corrected_pvals, _, _ = multipletests(p_values, alpha=0.05, method='holm')

    print("\nPaired t-tests with Holm–Bonferroni correction:")
    for i, label in enumerate(test_labels):
        sig = "✅" if reject[i] else "❌"
        print(f"{label}: p = {p_values[i]:.4e}, corrected = {corrected_pvals[i]:.4e} {sig}")

    # Bar plot with significance indicators
    means_1, lower_1, upper_1 = zip(*[mean_ci(per_structure_mrr_1[s]) for s in query_structures])
    means_2, lower_2, upper_2 = zip(*[mean_ci(per_structure_mrr_2[s]) for s in query_structures])
    means_combined, lower_c, upper_c = zip(*[mean_ci(per_structure_mrr_combined[s]) for s in query_structures])

    ci_1 = [ [m-l, u-m] for m, l, u in zip(means_1, lower_1, upper_1)]
    ci_2 = [ [m-l, u-m] for m, l, u in zip(means_2, lower_2, upper_2)]
    ci_combined = [ [m-l, u-m] for m, l, u in zip(means_combined, lower_c, upper_c)]

    x = range(len(query_structures))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4))
    bars1 = ax.bar([i - width for i in x], means_1, width, yerr=np.array(ci_1).T, label=f"{model_name_1}", color='#a3cbe6ff', capsize=5)
    bars2 = ax.bar(x, means_2, width, yerr=np.array(ci_2).T, label=f"{model_name_2}", color='#6cacd6ff', capsize=5)
    bars3 = ax.bar([i + width for i in x], means_combined, width, yerr=np.array(ci_combined).T, label="Combined", color='#3280b5ff', capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(query_structures)
    ax.set_ylabel('Mean MRR')
    ax.set_xlabel('Query type')

    pprint(upper_1)

    # Add significance indicators
    for i, s in enumerate(query_structures):
        # Find p-values for this structure
        idx1 = test_labels.index(f"{s}: model_1 vs combined") if f"{s}: model_1 vs combined" in test_labels else None
        idx2 = test_labels.index(f"{s}: model_2 vs combined") if f"{s}: model_2 vs combined" in test_labels else None
        # Use the max upper CI for this structure

        y_max = max(upper_1[i], upper_2[i], upper_c[i])
        y_offset = 0.02
        h = y_max + y_offset
        if idx1 is not None:
            p_val = corrected_pvals[idx1]
            sig = "*" if p_val < 0.05 else "ns"
            x1 = i - width
            x2 = i + width
            y = h
            ax.plot([x1, x1, x2, x2], [y, y + y_offset, y + y_offset, y], lw=1.5, c='k')
            ax.text(i, y + y_offset, f"{significance_marker(p_val)}", ha='center', va='bottom', fontsize=10)
            h += y_offset * 4
        if idx2 is not None:
            p_val = corrected_pvals[idx2]
            sig = "*" if p_val < 0.05 else "ns"
            x1 = i
            x2 = i + width
            y = h
            ax.plot([x1, x1, x2, x2], [y, y + y_offset, y + y_offset, y], lw=1.5, c='k')
            ax.text(i + width/2, y + y_offset, f"{significance_marker(p_val)}", ha='center', va='bottom', fontsize=10)
            h += y_offset * 2

    ax.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mrr1', type=str, required=True, help='Pickle file for per_structure_mrr_1')
    parser.add_argument('--mrr2', type=str, required=True, help='Pickle file for per_structure_mrr_2')
    parser.add_argument('--mrr_combined', type=str, required=True, help='Pickle file for per_structure_mrr_combined')
    parser.add_argument('--model_1', type=str, required=True, help='Name of model 1')
    parser.add_argument('--model_2', type=str, required=True, help='Name of model 2')
    parser.add_argument('--query_structures', type=str, help='Comma-separated list of query structures', default="1p,2p,3p,2i,3i,ip,pi")
    args = parser.parse_args()
    main(args)