# SVR baselines

Support-vector regression with an RBF kernel (`C=10000`, `epsilon=0.1`, no feature scaling) on four count-descriptor sets:

| Set | Features | Scripts |
|---|---|---|
| atom | 30 atom-type counts | `SVR_predS_atom*.py` |
| element | 13 element totals | `SVR_predS_element*.py` |
| bond | 256 features (general descriptors, bond-type counts, symmetry) | `SVR_predS_bond*.py` |
| all | 289 features (atom, element and bond) | `SVR_predS_all*.py` |

## Files

| File | What it does |
|---|---|
| `SVR_predS_<set>.py` | Trains on a random 80% of `hybrid_train_091926.csv` (`random_state=42`), saves the model as `sv_<set>.pkl`, scores the other 20%, and writes `sv_<set>_results.csv`. |
| `SVR_predS_<set>2.py` | Loads a saved model and scores an external set (`hybrid_pdbbind.csv` or `hybrid_std6074.csv`). |
| `SVR_predS_all_grid.py` | 5-fold grid search over `C`, `epsilon` and the kernel. Needs `all_train_v2.csv`, which is not included. |
| `sv_*.pkl` | Saved models for the atom, element, bond and all sets. |
| `sv_*_results*.csv` | Predictions. `entropy` is the true value and `Pre` is the prediction. |
| `hybrid_*.csv` | Copies of the training and external CSVs. |

## Usage

```bash
cd SVR_general
python SVR_predS_bond.py
```

Before running, replace `/4T/projects/entropy/SVR_V2/` with `./` in the scripts. They save and load models at that path, which exists only on the original machine.

Notes:
- The scripts call `mean_squared_error(..., squared=False)`, which scikit-learn removed in version 1.6. Use scikit-learn before 1.6, or replace it with `np.sqrt(mean_squared_error(...))`.
- The scripts end with `plt.show()`, which needs a display.
