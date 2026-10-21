#!/usr/bin/env python3
"""Calculate R2, MAE, RMSE, Pearson correlation, and MRE from a prediction CSV.

Expected columns: obs, pred (override with --obs / --pred).
"""
print("USUAGE:python3 calc_metrics.py -i ./sv_bond_results_std109.csv --obs obs --pred pred\n python3 calc_metrics.py --input ./another_results.csv --obs obs --pred pred")

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def mean_relative_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if np.any(y_true == 0):
        raise ValueError("y_true contains zero values, cannot compute relative error.")
    return np.mean(np.abs(y_true - y_pred) / y_true)


def main():
    parser = argparse.ArgumentParser(description="Compute regression metrics from a prediction CSV.")
    parser.add_argument(
        "-i",
        "--input",
        dest="csv_path",
        required=True,
        help="Input CSV with observed and predicted columns",
    )
    parser.add_argument("--obs", default="obs", help="Observed-value column name (default: obs)")
    parser.add_argument("--pred", default="pred", help="Predicted-value column name (default: pred)")
    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)
    y_true = data[args.obs]
    y_pred = data[args.pred]

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred,squared=False)
    mre = mean_relative_error(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)

    print("R2:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MRE:", mre)
    print("Pearsons correlation: %.4f" % corr)


if __name__ == "__main__":
    main()
