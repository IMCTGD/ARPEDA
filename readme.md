# ARPEDA: Attribute-Rank-Preserving Evolutionary Discretization Algorithm

This repository contains the main implementation of ARPEDA used for the UCI and medical-data experiments.

## Files

- `main_MOEA_D.py`: main program for the UCI benchmark experiments.
- `main_MOEA_D_medical_holdout.py`: main program for the sMRI and EEG experiments.
- `module/`: functions required by the two main programs.
- `uci_result/`: generated results for the UCI benchmark datasets.
- `medical_result/`: generated results for the sMRI and EEG datasets.

## Dataset
The dataset is hosted on Baidu Netdisk:
- Share link:  https://pan.baidu.com/s/1r10zLy1auvbj7OFsshcTfQ?pwd=cbee
- Code: cbee


## Requirements

The programs require Python 3 and the following packages:

`numpy`, `pandas`, `scipy`, `scikit-learn`, `catboost`, `deap`, and `joblib`.

## Running the programs

Run the UCI experiments with:

```bash
python main_MOEA_D.py
```


Run the medical independent-test experiments with:

```bash
python main_MOEA_D_medical_holdout.py --data_dir <data-root> --output_dir <output-directory> --protocol holdout
```
