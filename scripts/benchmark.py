"""Benchmark — measure Levenshtein score against a ground truth CSV."""

import argparse

import pandas as pd


def levenshtein(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def score(pred_path: str, truth_path: str) -> None:
    pred = pd.read_csv(pred_path).set_index("id")
    truth = pd.read_csv(truth_path).set_index("id")

    common = pred.index.intersection(truth.index)
    total_dist = 0
    worst: list[tuple[int, str]] = []

    for id_ in common:
        p = str(pred.loc[id_, "votes"])
        t = str(truth.loc[id_, "votes"])
        d = levenshtein(p, t)
        total_dist += d
        if d > 0:
            worst.append((d, id_))

    worst.sort(reverse=True)
    print(f"\nTotal Levenshtein distance : {total_dist}")
    print(f"Rows compared              : {len(common)}")
    print(f"Perfect rows               : {len(common) - len(worst)}")
    if worst:
        print(f"\nTop-10 worst predictions:")
        for dist, id_ in worst[:10]:
            p = str(pred.loc[id_, "votes"])
            t = str(truth.loc[id_, "votes"])
            print(f"  {id_:45s}  pred={p!r:>12}  truth={t!r:>12}  dist={dist}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred", help="submission CSV to evaluate")
    parser.add_argument("truth", help="ground truth CSV")
    args = parser.parse_args()
    score(args.pred, args.truth)
