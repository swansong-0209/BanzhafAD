# BanzhafAD: Cooperative Banzhaf Interaction Modeling for Interpretable Multivariate Time-Series Anomaly Detection

## Get Started

### 1. Dataset

The datasets used in this project are sourced from the
[TSB-AD](https://github.com/TheDatumOrg/TSB-AD) benchmark collection.

After downloading the datasets, place the dataset files in the following
directory:

```text
BanzhafAD/Datasets/
```

When working from the root directory of this repository, the corresponding
relative path is:

```text
Datasets/
```

The expected project structure is:

```text
BanzhafAD/
├── Datasets/
├── run_all_models_TSB_AD_aligned.py
├── TSB_AD/
├── requirements.txt
└── ...
```

The datasets are used for research, benchmarking, and reproducibility
purposes. Although TSB-AD provides a curated benchmark collection, the
copyright, license terms, citation requirements, and usage restrictions of
each dataset remain with their respective original providers.

Before redistributing a dataset or using it for commercial purposes, please
review:

* The original dataset source.
* The dataset-specific license.
* The required academic citations.
* Any redistribution or commercial-use restrictions.

When using these datasets in academic work, please cite both the relevant
original dataset sources and the TSB-AD paper.

### 2. Installation

#### Step 1: Enter the Project Directory

From the directory containing the cloned repository, enter the project
directory:

```bash
cd BanzhafAD
```

#### Step 2: Create a Conda Environment

Create a Conda environment with Python 3.11:

```bash
conda create -n banzhafad python=3.11 -y
```

Activate the environment:

```bash
conda activate banzhafad
```

#### Step 3: Install Dependencies

Install PyTorch with CUDA 12.1 support:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions or installation methods, refer to the
[PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/).

Install the remaining required Python packages:

```bash
pip install scikit-learn tqdm pandas statsmodels
```

To verify that PyTorch and CUDA are available, run:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```


#### Step 4: Run the Project

运行代码：

```bash
python run_all_models_TSB_AD_aligned.py
```

Make sure that the datasets have been placed in the `Datasets/` directory
before running the script.

#### Step 5: Remove the Conda Environment

Deactivate the environment:

```bash
conda deactivate
```

Remove the environment when it is no longer needed:

```bash
conda env remove -n banzhafad -y
```

## Acknowledgements

This project is built upon and includes code adapted from
[TSB-AD](https://github.com/TheDatumOrg/TSB-AD), a benchmark framework for
time-series anomaly detection developed by
[TheDatumOrg](https://github.com/TheDatumOrg) and its contributors.

TSB-AD is distributed under the
[Apache License 2.0](https://github.com/TheDatumOrg/TSB-AD/blob/main/LICENSE).

The original TSB-AD authors and contributors retain copyright over the
original portions of the software. Any modifications and additional code in
this repository are maintained by the BanzhafAD authors and contributors.

This project is independently maintained and is not affiliated with,
sponsored by, or endorsed by TheDatumOrg or the original TSB-AD authors.

For detailed third-party attribution and licensing information, see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

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
