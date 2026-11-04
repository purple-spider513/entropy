# Prediction speed test (10k to 500k molecules)

Times `svr_general_residual_unimol_predict.py` with the all-features residual model (`../unimol_residual/svr_general_residual_run_all_conformer_symmetry`) on unlabelled sets of 10,000 to 500,000 molecules. Every molecule in these sets has 25 heavy atoms. Each set ran in a fresh process with a fresh output folder, so no Uni-Mol conformers were reused between runs. The sets ran one after another on one RTX 5070 Ti.

## Results

| Molecules | Time (s) | Time | Molecules per second |
|---|---|---|---|
| 10,000 | 108 | 1 min 48 s | 93 |
| 50,000 | 510 | 8 min 30 s | 98 |
| 100,000 | 1,010 | 16 min 50 s | 99 |
| 200,000 | 2,008 | 33 min 28 s | 100 |
| 500,000 | 5,021 | 1 h 23 min 41 s | 100 |

Time grows linearly with the number of molecules, at about 10 ms per molecule. Most of it is Uni-Mol conformer generation and inference; the SVR stage is fast.

## Files

| File | Content |
|---|---|
| `run_speed_test_all.sh` | The timing script. |
| `timing.txt` | Wall-clock seconds per set. `driver.log` is a copy of the same output. |
| `predict_<n>.log` | Output of each prediction run. |
| `eval_<n>/hybrid_test_<n>_preds.csv` | Predictions, one row per molecule. **Use `Predicted_Entropy`** (SVR + Uni-Mol residual); `SVRPred` is the SVR stage alone. |
| `eval_<n>/metrics.json` | Written by the prediction script. **Not meaningful here**: these sets have no labels, so the `Entropy` column is 0 for every molecule and the MAE equals the mean prediction. |
| `eval_<n>/hybrid_test_<n>_conformers/test.predict.0.csv` | Raw Uni-Mol residual predictions. The `unimol_conformers.sdf` files (190 MB to 1.9 GB) are not in the repository. |

## Rerun

The input files `hybrid_test_10k.csv` to `hybrid_test_500k.csv` are not included. To rerun, place them in `unimol_residual/hybrid_data/` (they need `ID`, `SMILES` and the 289 descriptor columns), download the all-features checkpoint (see the top-level README), and run from the repository root:

```bash
GPU=0 bash eval_500k/run_speed_test_all.sh
```

Results go to `eval_500k_rerun/`, so the included results are not overwritten. Set `OUT=...` to choose another folder. Timings depend on the GPU and CPU.
