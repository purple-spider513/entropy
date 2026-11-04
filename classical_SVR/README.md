# SVR baselines

Support-vector regression with an RBF kernel (`C=10000`, `epsilon=0.1`, no feature scaling) on four count-descriptor sets:

| Set | Features | Scripts |
|---|---|---|
| atom | 35: 30 atom-type counts + 5 general descriptors | `SVR_predS_atom*.py` |
| element | 18: 13 element totals + 5 general descriptors | `SVR_predS_element*.py` |
| bond | 256: 250 bond-type counts + 6 general descriptors | `SVR_predS_bond*.py` |
| all | 289: general descriptors, element totals, atom types and bond types | `SVR_predS_all*.py` |

The general descriptors are `ring`, `heavy`, `s_heavy_bond`, `MW` and `symmetry` (the conformer and rotational-symmetry entropy correction). The bond set also has the hydrogen count `H` as a sixth general descriptor; in the atom and element sets, `H` is one of the counts. The bond set is the same 256 columns as `selected_features` in the other model folders. The saved `sv_*.pkl` models were trained on these default sets.

This folder also holds the training and external test CSVs used by every model in the repository (see the top-level README).

## Files

| File | What it does |
|---|---|
| `SVR_predS_<set>.py` | Trains on a random 80% of `hybrid_train_102926.csv` (`random_state=42`), saves the model as `sv_<set>.pkl`, scores the other 20%, and writes `sv_<set>_results.csv`. |
| `SVR_predS_<set>2.py` | Loads a saved model and scores an external set (`hybrid_pdbbind_102926.csv` or `hybrid_std6074_102926.csv`). Edit the file name in the script to score another set. |
| `sv_*.pkl` | Saved models for the atom, element, bond and all sets. |
| `sv_*_results*.csv` | Predictions. `entropy` is the true value and `Pre` is the prediction. |
| `hybrid_*_102926.csv` | Training and external test CSVs. |

## Choosing the features

Each script picks its feature columns in one line near the top: `X=data[[...]]` in the training scripts and `X_test=test_data[[...]]` in the `*2.py` scripts. The atom, element and bond scripts also have a second, commented-out line right below it, with only the counts and no general descriptors:

| Set | Active line (default) | Commented-out line |
|---|---|---|
| atom | 35 columns | 30 atom-type counts, including `H` |
| element | 18 columns | 13 element totals |
| bond | 256 columns | 250 bond-type counts |

`SVR_predS_all*.py` has only one line.

To train on the counts only, put `#` in front of the active line and remove the `#` from the other one:

```python
#X=data[["ring","heavy","H","s_heavy_bond","MW", ... ,"symmetry"]]   # default, now off
X=data[["B-C.3","B-C.ar","B-O.3", ... ]]                             # counts only, now on
```

Keep exactly one `X=` line active. Any other list of CSV column names works too.

A model only accepts the columns it was trained on, in the same order. If you change the list in `SVR_predS_<set>.py`, make the same change in `SVR_predS_<set>2.py` and retrain before scoring external sets. Retraining overwrites `sv_<set>.pkl`, so copy the original first if you want to keep it, or change the `.pkl` file name in both scripts.

## Usage

```bash
cd classical_SVR
python SVR_predS_bond.py
python SVR_predS_bond2.py
```

Notes:
- The scripts call `mean_squared_error(..., squared=False)`, which scikit-learn removed in version 1.6. The script then stops after writing its predictions CSV, before printing RMSE. Replace that call with `np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_pre))` (the scripts already import numpy as `np`).
- The saved `sv_*.pkl` models were written with scikit-learn 1.8. Load them with 1.8 or newer.
- The scripts end with `plt.show()`, which needs a display.
