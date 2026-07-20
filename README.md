# BanzhafAD
BanzhafAD: Cooperative Banzhaf Interaction Modeling for Interpretable Multivariate Time-Series Anomaly Detection


## Get Started

### 1. Dataset

The datasets used in this project were obtained through the
[TSB-AD](https://github.com/TheDatumOrg/TSB-AD) benchmark collection.

The dataset files are stored in:

```text
BanzhafAD/Datasets/
```

You can access them from the repository through:

[`BanzhafAD/Datasets`](BanzhafAD/Datasets)

The datasets are included for research, benchmarking, and reproducibility
purposes. Although TSB-AD provides a curated benchmark collection, the
copyright, license terms, citation requirements, and usage restrictions of
each dataset remain with their respective original providers.

Before redistributing the datasets or using them for commercial purposes,
please review:

* The original dataset source.
* The dataset-specific license.
* The required academic citations.
* Any redistribution or commercial-use restrictions.

When using these datasets in academic work, please cite both the relevant
original dataset sources and the TSB-AD paper.

### 2. Installation

#### Step 1: Create a Conda Environment

Create a Python 3.11 virtual environment:

```bash
conda create -p /root/BanzhafAD python=3.11 -y
```

Activate the environment:

```bash
conda activate /root/BanzhafAD
```

#### Step 2: Enter the Project Directory

Change to the project source directory:

```bash
cd /root/TSB-AD-main
```

#### Step 3: Install Dependencies

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Install the project in editable mode:

```bash
pip install -e .
```

#### Step 4: Run the Project

Run the default time-series anomaly detection example:

```bash
python run_all_models_TSB_AD_aligned.py
```




## Acknowledgements

This project is built upon and includes code adapted from
[TSB-AD](https://github.com/TheDatumOrg/TSB-AD), a benchmark framework for
time-series anomaly detection developed by
[TheDatumOrg](https://github.com/TheDatumOrg) and its contributors.

TSB-AD is distributed under the
[Apache License 2.0](https://github.com/TheDatumOrg/TSB-AD/blob/main/LICENSE).

The original TSB-AD authors and contributors retain copyright over the
original portions of the software. 

This project is independently maintained and is not affiliated with, sponsored
by, or endorsed by TheDatumOrg or the original TSB-AD authors.



## Citation

If you use this project in academic work, please also cite the original TSB-AD
paper:

```bibtex
@inproceedings{liu2024elephant,
  title     = {The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark},
  author    = {Liu, Qinghua and Paparrizos, John},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

