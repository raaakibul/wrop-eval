"""
validate.py — correlate per-model OPS (fully automatic) against human
blind pairwise Elo ratings (expensive, slow, one-off).  
"""
from __future__ import annotations
from typing import Dict
import numpy as np
from scipy import stats
import json
import sys
from typing import Dict


def correlate(ops: Dict[str, float], elo: Dict[str, float]) -> dict:
    common = sorted(set(ops) & set(elo))
    if len(common) < 3:
        raise ValueError(
            f"Need >=3 models present in both score dicts, got {len(common)}: {common}"
        )
    x = np.array([ops[m] for m in common])
    y = np.array([elo[m] for m in common])
    spearman = stats.spearmanr(x, y)
    pearson = stats.pearsonr(x, y)

    # Rank disagreement
    # rank, sorted from most to least disagreement.
    ops_rank = stats.rankdata(-x)
    elo_rank = stats.rankdata(-y)
    residuals = sorted(
        zip(common, ops_rank - elo_rank),
        key=lambda kv: -abs(kv[1]),
    )

    return {
        "n_models": len(common),
        "spearman_r": float(spearman.correlation),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "models": common,
        "ops": [float(v) for v in x],
        "elo": [float(v) for v in y],
        "rank_disagreement_most_to_least": [
            {"model": m, "ops_rank_minus_elo_rank": float(d)} for m, d in residuals
        ],
    }
    

def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        print("usage: python -m wrop_eval.validate ops_per_model.json elo_per_model.json")
        raise SystemExit(2)
    with open(argv[0]) as f:
        ops = json.load(f)
    with open(argv[1]) as f:
        elo = json.load(f)
    result = correlate(ops, elo)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()