import time
import os
import pandas as pd
import numpy as np
import re
import torch
import warnings
import random
from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD, Unsupervise_AD_Pool, Semisupervise_AD_Pool
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict, Optimal_Uni_algo_HP_dict
from TSB_AD.evaluation.metrics import get_metrics
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ====================  Academic Experiment Configurations ====================
# Task Type Configuration: Select 'TSB-AD-M' (Multivariate) or 'TSB-AD-U' (Univariate)
TASK_TYPE = 'TSB-AD-M'

# Fixed stage as EVAL for reporting final academic benchmark results with locked parameters
CURRENT_STAGE = 'EVAL'


if TASK_TYPE == 'TSB-AD-M':
    HP_DICT_POOL = Optimal_Multi_algo_HP_dict
    DATASET_DATA_DIR = 'Datasets/TSB-AD-M'
    FILE_LIST_PATH = 'Datasets/File_List/TSB-AD-M-Eva.csv'
elif TASK_TYPE == 'TSB-AD-U':
    HP_DICT_POOL = Optimal_Uni_algo_HP_dict
    DATASET_DATA_DIR = 'Datasets/TSB-AD-U'
    FILE_LIST_PATH = 'Datasets/File_List/TSB-AD-U-Eva.csv'
else:
    raise ValueError(f"Unknown task type: {TASK_TYPE}. Please check configuration.")


def build_complete_dataset_configs(file_list_path):
    if not os.path.exists(file_list_path):
        return {}
    try:
        df_list = pd.read_csv(file_list_path)
        if 'file_name' not in df_list.columns:
            return {}

        detected_keywords = set()
        for fname in df_list['file_name'].dropna():
            parts = fname.split('_')
            if len(parts) > 1:
                keyword = parts[1] if parts[0].isdigit() else parts[0]
                detected_keywords.add(keyword)
            else:
                detected_keywords.add(os.path.splitext(fname)[0])

        configs = {}
        for kw in sorted(detected_keywords):
            configs[kw] = {
                'keyword': kw,
                'description': f'TSB-AD Sub-Dataset: {kw}',
            }
        return configs
    except Exception as e:
        print(f"Failed to dynamically parse File_List ({e}). Falling back to empty configuration.")
        return {}


# Generate dataset configurations
DATASET_CONFIGS = build_complete_dataset_configs(FILE_LIST_PATH)

# Filter out algorithms missing locally or guaranteed to throw errors
BLACK_LIST = ['NormA', 'Series2Graph', 'CHARM', 'SR', 'FFT']

# Target model and datasets selection
TARGET_MODEL = 'BanzhafAD'
TARGET_DATASETS = ['MSL', 'PSM', 'SWaT', 'GHL', 'Genesis', 'LTDB']  # 'None' to run all datasets

# Seed settings for repeatable evaluation
SEEDS = [2024, 42, 2021]
N_REPEATS = len(SEEDS)


# ===========================================================
def get_dataset_entities_from_list(file_list_path, data_dir, config):
    """
    Parse entity file paths strictly based on the official File_List
    """
    if not os.path.exists(file_list_path):
        raise FileNotFoundError(f"Official File_List config file not found: '{file_list_path}'")

    df_list = pd.read_csv(file_list_path)
    if 'file_name' not in df_list.columns:
        raise ValueError(f"Invalid format in config file {file_list_path}, missing 'file_name' column.")

    keyword = config['keyword']

    def _extract_dataset_name(fname):
        if not isinstance(fname, str):
            return None
        parts = fname.split('_')
        if len(parts) > 1:
            return parts[1] if parts[0].isdigit() else parts[0]
        return os.path.splitext(fname)[0]

    filtered_df = df_list[df_list['file_name'].apply(_extract_dataset_name) == keyword]

    files = []
    missing_files = []
    official_expected_list = filtered_df['file_name'].tolist()

    for fname in official_expected_list:
        full_path = os.path.join(data_dir, fname)
        if os.path.exists(full_path):
            files.append(full_path)
        else:
            missing_files.append(fname)

    if CURRENT_STAGE == 'EVAL' and missing_files:
        raise FileNotFoundError(
            f" [PROTOCOL VIOLATION] Official evaluation stage interception: Missing official entity files locally!\n"
            f"   Expected count: {len(official_expected_list)} | Found count: {len(files)}\n"
            f"   Missing files: {missing_files}"
        )
    elif missing_files:
        print(f"Warning (TUNING Mode): Some files in official list "
              f"were not found locally and skipped:\n   {missing_files}")

    if not files:
        raise FileNotFoundError(
            f" No matching entity files found under '{data_dir}' for keyword '{keyword}'!"
        )
    return sorted(files), len(official_expected_list)


def load_single_entity(path):
    """
    Adaptive academic split following TSB-AD file naming conventions.
    """
    df = pd.read_csv(path).dropna()

    if 'Label' not in df.columns:
        raise KeyError(f" Official 'Label' column not found in data file: {path}")
    y = df['Label'].astype(int).to_numpy()

    X = df.drop(columns=['Label']).values.astype(float)

    file_name = os.path.basename(path)
    tr_match = re.search(r'_tr_(\d+)', file_name)

    if not tr_match:
        raise ValueError(
            f" Protocol validation failed: Unable to parse official '_tr_' training split index from '{file_name}'!"
        )

    split_idx = int(tr_match.group(1))
    if split_idx >= len(X):
        raise ValueError(
            f" Protocol validation failed: Specified train set length '{split_idx}' "
            f"exceeds or equals sequence length '{len(X)}'!"
        )

    data_train = X[:split_idx]

    return X, y, data_train


def run_single_experiment(file_list, all_models, round_idx):
    """Execute a single round of evaluation and return Macro-Mean metric dictionaries and failure logs."""
    print(f"\n{'#' * 50} Round {round_idx}/{N_REPEATS} Experiment {'#' * 50}")

    from TSB_AD.utils.slidingWindows import find_length_rank

    model_raw_results = {m[1]: [] for m in all_models}
    model_failure_log = {m[1]: [] for m in all_models}

    for f_idx, file_path in enumerate(file_list, 1):
        file_name = os.path.basename(file_path)
        print(f"\n === [Entity {f_idx}/{len(file_list)}] Processing: {file_name} ===")

        try:
            X_full, y_full, data_train = load_single_entity(file_path)

            slidingWindow = find_length_rank(X_full[:, 0].reshape(-1, 1), rank=1)

            print(
                f"    Dimension inspection -> Length: {X_full.shape[0]}, "
                f"Features: {X_full.shape[1]} | Window (slidingWindow): {slidingWindow}")

        except Exception as e:
            print(f"    Failed to load file {file_name}, skipping. Reason: {e}")
            continue

        for m_idx, (mode, model_name) in enumerate(all_models, 1):
            try:
                if model_name not in HP_DICT_POOL:
                    raise KeyError(
                        f"Protocol Block: Algorithm '{model_name}' not found "
                        f"in official optimal hyperparameters pool for [{TASK_TYPE}]!"
                    )

                current_hp = HP_DICT_POOL[model_name]

                start_time = time.time()
                if mode == 'unsupervised':
                    output = run_Unsupervise_AD(model_name, X_full, **current_hp)
                    current_y = y_full
                else:
                    output = run_Semisupervise_AD(model_name, data_train, X_full, **current_hp)
                    current_y = y_full
                elapsed_time = time.time() - start_time

                if isinstance(output, str) or output is None:
                    raise ValueError("Model returned invalid output score.")

                metrics = get_metrics(output, current_y, slidingWindow=slidingWindow)

                # Follow the official TSB-AD clean evaluation protocol
                res_dict = {
                    "Time(s)": elapsed_time,  # Pure model training and inference time
                    "ROC-AUC": metrics.get('AUC-ROC', 0.0),
                    "PR-AUC": metrics.get('AUC-PR', 0.0),
                    "VUS-ROC": metrics.get('VUS-ROC', 0.0),
                    "VUS-PR": metrics.get('VUS-PR', 0.0),
                    "Point-F1": metrics.get('Standard-F1', 0.0),
                    "PA-F1": metrics.get('PA-F1', 0.0),
                    "Event-F1": metrics.get('Event-based-F1', 0.0),
                    "Range-F1": metrics.get('R-based-F1', 0.0),
                    "Affiliation-F": metrics.get('Affiliation-F', 0.0)
                }
                model_raw_results[model_name].append(res_dict)
                print(f"  [{m_idx}/{len(all_models)}] {model_name}: Success | VUS-PR: {res_dict['VUS-PR']:.4f}")

            except Exception as e:
                full_reason = str(e)
                model_failure_log[model_name].append((file_name, full_reason))
                print(f"  [{m_idx}/{len(all_models)}] {model_name}: Skipped. Reason: {full_reason}")
                continue
            finally:
                import gc
                torch.cuda.empty_cache()
                gc.collect()

    round_report = {}
    for model_name, results in model_raw_results.items():
        if not results:
            continue
        df_model_perf = pd.DataFrame(results)
        mean_perf = df_model_perf.mean()
        round_report[model_name] = {
            "Success_Files": len(results),
            "Time(s)": mean_perf["Time(s)"],
            "ROC-AUC": mean_perf["ROC-AUC"],
            "PR-AUC": mean_perf["PR-AUC"],
            "VUS-ROC": mean_perf["VUS-ROC"],
            "VUS-PR": mean_perf["VUS-PR"],
            "Point-F1": mean_perf["Point-F1"],
            "PA-F1": mean_perf["PA-F1"],
            "Event-F1": mean_perf["Event-F1"],
            "Range-F1": mean_perf["Range-F1"],
            "Affiliation-F": mean_perf["Affiliation-F"]
        }

    return round_report, model_failure_log


def main():
    global DATASET_CONFIGS
    DATASET_CONFIGS = build_complete_dataset_configs(FILE_LIST_PATH)

    print("-" * 30)
    print(f" Academic Evaluation Protocol Check | Mode: [PURE EVAL RUNNER]")
    print(f" Loading Official Evaluation File List: {FILE_LIST_PATH}")
    print("-" * 30)

    all_models = [('unsupervised', m) for m in Unsupervise_AD_Pool if m not in BLACK_LIST] + \
                 [('semisupervised', m) for m in Semisupervise_AD_Pool if m not in BLACK_LIST]

    if TARGET_MODEL is not None:
        all_models = [(mode, name) for mode, name in all_models if name == TARGET_MODEL]
        if not all_models:
            print(f" Specified model '{TARGET_MODEL}' is not in the available model pool!")
            print(f"   Note: Check if '{TARGET_MODEL}' is registered in Unsupervise_AD_Pool "
                  f"or Semisupervise_AD_Pool in TSB_AD.model_wrapper.")
            return
        print(f" Single-model mode: Running [{TARGET_MODEL}] only (Detected mode: {all_models[0][0]})")

    for dataset_name, ds_config in DATASET_CONFIGS.items():
        if TARGET_DATASETS is not None and dataset_name not in TARGET_DATASETS:
            continue

        print(f"\n\n" + "=" * 115)
        print(f"  Current Evaluation Dataset: {ds_config['description']}")
        print(f"   Official Metadata List: {FILE_LIST_PATH} | Keyword Filter: {ds_config['keyword']} "
              f"| Window: [TSB-AD Official Dynamic Estimation]")
        print(f"   Repeat Count: {N_REPEATS}")
        print("=" * 115)

        try:
            file_list, official_expected_count = get_dataset_entities_from_list(FILE_LIST_PATH, DATASET_DATA_DIR,
                                                                                ds_config)
            print(f" Official Expected Files: {official_expected_count} | Found & Loaded Locally: {len(file_list)}")
        except Exception as e:
            print(str(e))
            continue

        print(f"  Total Models to Evaluate: {len(all_models)}")
        print(f"  Running {N_REPEATS} rounds over {len(file_list)} files to produce Mean ± Std final report.")
        print("-" * 115)

        all_rounds_results = {m[1]: [] for m in all_models}
        all_rounds_failures = {m[1]: {} for m in all_models}

        for round_idx in range(1, N_REPEATS + 1):
            current_seed = SEEDS[round_idx - 1]
            random.seed(current_seed)
            np.random.seed(current_seed)
            if torch is not None:
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(current_seed)

            print(f"  [Round {round_idx}] Random Seed set to: {current_seed}")
            round_report, failure_log = run_single_experiment(file_list, all_models, round_idx)

            for model_name, perf in round_report.items():
                all_rounds_results[model_name].append(perf)

            for model_name, perf in round_report.items():
                print(f"\n{'─' * 50}")
                print(
                    f"   Model: {model_name}  |  Round {round_idx} Single-Round Mean Results  "
                    f"|  Success Files: {perf['Success_Files']}")
                print(f"{'─' * 50}")
                print(f"  Time(s) (Pure): {perf['Time(s)']:.4f}")
                print(f"  VUS-PR       : {perf['VUS-PR']:<22.4f}  VUS-ROC      : {perf['VUS-ROC']:.4f}")
                print(f"  ROC-AUC      : {perf['ROC-AUC']:<22.4f}  PR-AUC       : {perf['PR-AUC']:.4f}")
                print(f"  Point-F1     : {perf['Point-F1']:<22.4f}  PA-F1        : {perf['PA-F1']:.4f}")
                print(f"  Event-F1     : {perf['Event-F1']:<22.4f}  Range-F1     : {perf['Range-F1']:.4f}")
                print(f"  Affiliation-F: {perf['Affiliation-F']:.4f}")
                print(f"{'═' * 60}\n")
            # ==============================================================================

            # Log the details of files that failed in the current iteration
            for model_name, logs in failure_log.items():
                if logs:
                    all_rounds_failures[model_name][round_idx] = logs

        # ==================== Final Summary: Mean ± Std Table for the Current Dataset ====================
        print("\n" + "★" * 60)
        print(f"★    Final Summary: {dataset_name} | {N_REPEATS} Rounds Experiment Mean ± Std   ")
        print(f"★  Official Expected Files: {official_expected_count} | Found Local Files: {len(file_list)}")
        print("★" * 60)

        metric_cols = ["Time(s)", "ROC-AUC", "PR-AUC", "VUS-ROC", "VUS-PR",
                       "Point-F1", "PA-F1", "Event-F1", "Range-F1", "Affiliation-F"]

        summary_rows = []
        dataset_is_incomplete = len(file_list) < official_expected_count

        for model_name, rounds in all_rounds_results.items():
            if not rounds:
                continue

            df_rounds = pd.DataFrame(rounds)
            is_incomplete = dataset_is_incomplete or any(r["Success_Files"] < official_expected_count for r in rounds)
            display_name = f"{model_name} [INCOMPLETE_DATASET]" if is_incomplete else model_name

            row = {"Model": display_name, "Raw_Model_Name": model_name, "Rounds": len(rounds)}

            for col in metric_cols:
                mean_val = df_rounds[col].mean()
                std_val = df_rounds[col].std(ddof=1) if len(df_rounds) > 1 else 0.0
                row[col] = f"{mean_val:.4f} ± {std_val:.4f}"

            row["_sort_key"] = df_rounds["VUS-PR"].mean()
            summary_rows.append(row)

        if summary_rows:
            summary_rows.sort(key=lambda x: x["_sort_key"], reverse=True)

            for row in summary_rows:
                model_name = row["Model"]
                rounds = row["Rounds"]
                print(f"\n{'─' * 50}")
                print(f"  Model: {model_name}  |  Successful Rounds: {rounds}/{N_REPEATS}")
                print(f"{'─' * 50}")
                print(f"  Time(s) (Pure): {row['Time(s)']}")
                print(f"  VUS-PR ()   : {row['VUS-PR']:<22s}  VUS-ROC      : {row['VUS-ROC']}")
                print(f"  ROC-AUC      : {row['ROC-AUC']:<22s}  PR-AUC       : {row['PR-AUC']}")
                print(f"  Point-F1     : {row['Point-F1']:<22s}  PA-F1        : {row['PA-F1']}")
                print(f"  Event-F1     : {row['Event-F1']:<22s}  Range-F1     : {row['Range-F1']}")
                print(f"  Affiliation-F: {row['Affiliation-F']}")
        else:
            print(f" No algorithm successfully generated results for dataset {dataset_name}.")

    print("\n" + "=" * 135)
    print(" All datasets completed successfully!")


if __name__ == '__main__':
    main()
