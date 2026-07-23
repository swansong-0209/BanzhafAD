import time
import os
import pandas as pd
import numpy as np
import re
import torch
import warnings
import random
import gc
from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD, Unsupervise_AD_Pool, Semisupervise_AD_Pool
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict, Optimal_Uni_algo_HP_dict
from TSB_AD.evaluation.metrics import get_metrics
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============  Experiment Configuration ===============
# task type configuration: 'TSB-AD-M' (multivariate) or 'TSB-AD-U' (univariate)
TASK_TYPE = 'TSB-AD-M'

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
    raise ValueError(f"Unknown task type: {TASK_TYPE}, please check configuration.")

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
        print(f"️ Failed to dynamically parse File_List ({e}), falling back to empty config.")
        return {}


# Build complete dataset dictionary
DATASET_CONFIGS = build_complete_dataset_configs(FILE_LIST_PATH)

# Filter out algorithms that are missing locally or prone to errors
BLACK_LIST = ['NormA', 'Series2Graph', 'CHARM', 'SR', 'FFT']

# Specify target model and target datasets
TARGET_MODEL = 'BanzhafAD'    # 'BanzhafAD'
TARGET_DATASETS = ['MSL', 'PSM', 'SWaT', 'GHL', 'Genesis', 'LTDB']  # 'None' for all

# Specify seeds for repeated experiments
SEEDS = [2024, 42, 2021]
N_REPEATS = len(SEEDS)

# ===========================================================
def get_dataset_entities_from_list(file_list_path, data_dir, config):
    """
    Parse entity file paths based strictly on the official File_List.
    """
    if not os.path.exists(file_list_path):
        raise FileNotFoundError(f" Official File_List config file not found: '{file_list_path}'")

    df_list = pd.read_csv(file_list_path)
    if 'file_name' not in df_list.columns:
        raise ValueError(f" Config file {file_list_path} is invalid, missing 'file_name' column.")

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
            f" [PROTOCOL VIOLATION] Intercepted in EVAL stage: Missing official files locally!\n"
            f"   Expected count: {len(official_expected_list)} | Found count: {len(files)}\n"
            f"   Missing file list: {missing_files}"
        )
    elif missing_files:
        print(f"️ Warning (TUNING mode): Some official files were not found locally and skipped:\n   {missing_files}")

    if not files:
        raise FileNotFoundError(
            f" No matching entity files found in '{data_dir}' after filtering by keyword '{keyword}'!"
        )
    return sorted(files), len(official_expected_list)


def load_single_entity(path):
    """
    Load single data file adhering strictly to TSB-AD protocol.
    """
    df = pd.read_csv(path).dropna()

    if 'Label' not in df.columns:
        raise KeyError(f"Official 'Label' column not found in data file: {path}")
    y = df['Label'].astype(int).to_numpy()

    X = df.drop(columns=['Label']).values.astype(float)

    file_name = os.path.basename(path)
    tr_match = re.search(r'_tr_(\d+)', file_name)

    if not tr_match:
        raise ValueError(
            f"Protocol validation failed: Cannot parse '_tr_' train prefix index from filename '{file_name}'!"
        )

    split_idx = int(tr_match.group(1))
    if split_idx >= len(X):
        raise ValueError(
            f"Protocol validation failed: Train set split index '{split_idx}' exceeds data length '{len(X)}'!"
        )

    data_train = X[:split_idx]

    return X, y, data_train

def run_single_experiment(file_list, all_models, round_idx):
    """Run a single round of experiment and return Macro-Mean metrics dict and failure log per model"""
    print(f"\n{'#' * 50} Round {round_idx}/{N_REPEATS} {'#' * 50}")

    from TSB_AD.utils.slidingWindows import find_length_rank

    model_raw_results = {m[1]: [] for m in all_models}
    model_failure_log = {m[1]: [] for m in all_models}

    for f_idx, file_path in enumerate(file_list, 1):
        file_name = os.path.basename(file_path)
        print(f"\n === [Entity {f_idx}/{len(file_list)}] Processing: {file_name} ===")

        try:
            # start calculating the complete execution time
            total_start_time = time.time()

            X_full, y_full, data_train = load_single_entity(file_path)

            slidingWindow = find_length_rank(X_full[:, 0].reshape(-1, 1), rank=1)

            print(
                f"    Dimension inspection -> Length: {X_full.shape[0]}, Channels: {X_full.shape[1]} | Evaluation Window (slidingWindow): {slidingWindow}")
        except Exception as e:
            print(f"    Failed to load file {file_name}, skipped. Reason: {e}")
            continue

        for m_idx, (mode, model_name) in enumerate(all_models, 1):
            try:
                if model_name not in HP_DICT_POOL:
                    raise KeyError(
                        f"Protocol blocked: Algorithm '{model_name}' not found in optimal hyper-parameter dict for [{TASK_TYPE}]!"
                    )

                current_hp = HP_DICT_POOL[model_name]

                # Pure model execution timing (training and inference only)
                model_start_time = time.time()
                if mode == 'unsupervised':
                    output = run_Unsupervise_AD(model_name, X_full, **current_hp)
                    current_y = y_full
                else:
                    output = run_Semisupervise_AD(model_name, data_train, X_full, **current_hp)
                    current_y = y_full
                model_elapsed_time = time.time() - model_start_time

                if isinstance(output, str) or output is None:
                    raise ValueError("Model failed to return valid scores")

                # Metrics calculation performed after timing
                metrics = get_metrics(output, current_y, slidingWindow=slidingWindow)

                total_elapsed_time = time.time() - total_start_time

                # calculate target metrics
                res_dict = {
                    "Model_Time(s)": model_elapsed_time,  # Pure model execution timing
                    "Total_Time(s)": total_elapsed_time,  # total time consumption
                    "Affiliation-F1": metrics.get('Affiliation-F', 0.0),
                    "VUS-PR": metrics.get('VUS-PR', 0.0),
                    "Range-F1": metrics.get('R-based-F1', 0.0)
                }
                model_raw_results[model_name].append(res_dict)
                # print(f" [{m_idx}/{len(all_models)}] {model_name}: Success  | VUS-PR: {res_dict['VUS-PR']:.4f}")
                print(f" [{m_idx}/{len(all_models)}] {model_name}: Success | Affiliation-F1: {res_dict['Affiliation-F1']:.4f}")

            except Exception as e:
                full_reason = str(e)
                model_failure_log[model_name].append((file_name, full_reason))
                print(f"  [{m_idx}/{len(all_models)}] {model_name}: Skipped. Reason: {full_reason}")
                continue
            finally:
                torch.cuda.empty_cache()
                gc.collect()

    # Aggregate Macro-Mean results for current round
    round_report = {}
    for model_name, results in model_raw_results.items():
        if not results:
            continue
        df_model_perf = pd.DataFrame(results)
        mean_perf = df_model_perf.mean()
        round_report[model_name] = {
            "Success_Files": len(results),
            "Model_Time(s)": mean_perf["Model_Time(s)"],
            "Total_Time(s)": mean_perf["Total_Time(s)"],
            "Affiliation-F1": mean_perf["Affiliation-F1"],
            "VUS-PR": mean_perf["VUS-PR"],
            "Range-F1": mean_perf["Range-F1"]
        }

    return round_report, model_failure_log


def main():
    # Synchronize dataset configurations dynamically
    global DATASET_CONFIGS
    DATASET_CONFIGS = build_complete_dataset_configs(FILE_LIST_PATH)

    # Build model pool
    print("-" * 30)
    print(f" Academic Evaluation Protocol Check | Mode: [PURE EVAL RUNNER]")
    print(f" Loading official evaluation list: {FILE_LIST_PATH}")
    print("-" * 30)

    all_models = [('unsupervised', m) for m in Unsupervise_AD_Pool if m not in BLACK_LIST] + \
                 [('semisupervised', m) for m in Semisupervise_AD_Pool if m not in BLACK_LIST]

    if TARGET_MODEL is not None:
        all_models = [(mode, name) for mode, name in all_models if name == TARGET_MODEL]
        if not all_models:
            print(f" Specified model '{TARGET_MODEL}' is not available in model pool!")
            return
        print(f" Single-model mode: Running [{TARGET_MODEL}] only (Detected mode: {all_models[0][0]})")

    for dataset_name, ds_config in DATASET_CONFIGS.items():
        if TARGET_DATASETS is not None and dataset_name not in TARGET_DATASETS:
            continue

        print(f"\n\n" + "=" * 115)
        print(f"  Current Evaluation Dataset: {ds_config['description']}")
        print(f"   Official Metadata List: {FILE_LIST_PATH} | Keyword Filter: {ds_config['keyword']}")
        print(f"   Repeat Experiment Count: {N_REPEATS}")
        print("=" * 115)

        try:
            file_list, official_expected_count = get_dataset_entities_from_list(FILE_LIST_PATH, DATASET_DATA_DIR,
                                                                                ds_config)
            print(f" Official Expected Files: {official_expected_count} | Found & Loaded Locally: {len(file_list)}")
        except Exception as e:
            print(str(e))
            continue

        print(f"  Total algorithms to evaluate: {len(all_models)}")
        print(f"  Running {N_REPEATS} rounds across {len(file_list)} files, outputting final Mean ± Std table.")
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

            print(f"  [Round {round_idx}] Set random seed to: {current_seed}")
            round_report, failure_log = run_single_experiment(file_list, all_models, round_idx)

            for model_name, perf in round_report.items():
                all_rounds_results[model_name].append(perf)

            for model_name, perf in round_report.items():
                print(f"\n{'─' * 50}")
                print(
                    f"  Model: {model_name}  |  Round {round_idx} Single-Round Mean  |  Successful Files: {perf['Success_Files']}")
                print(f"{'─' * 50}")
                print(f"  Model_Time(s) : {perf['Model_Time(s)']:<22.4f} Total_Time(s) : {perf['Total_Time(s)']:.4f}")
                print(f"  Affiliation-F1: {perf['Affiliation-F1']:<22.4f} VUS-PR        : {perf['VUS-PR']:.4f}")
                print(f"  Range-F1      : {perf['Range-F1']:.4f}")
                print(f"{'═' * 60}\n")

            # Track file failure details for current round
            for model_name, logs in failure_log.items():
                if logs:
                    all_rounds_failures[model_name][round_idx] = logs

        # Final Summary: Mean ± Std across all rounds
        print("\n" + "-" * 60)
        print(f" Final Summary: {dataset_name} | {N_REPEATS} Rounds Mean ± Std   ")
        print(f" Official Expected Files: {official_expected_count} | Found Local Files: {len(file_list)}")
        print("" * 60)

        metric_cols = ["Model_Time(s)", "Total_Time(s)", "Affiliation-F1", "VUS-PR", "Range-F1"]

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
                print(f"  Model_Time(s) : {row['Model_Time(s)']:<26s} Total_Time(s) : {row['Total_Time(s)']}")
                print(f"  Affiliation-F1: {row['Affiliation-F1']:<26s} VUS-PR        : {row['VUS-PR']}")
                print(f"  Range-F1      : {row['Range-F1']}")
        else:
            print(f" No algorithm succeeded on dataset {dataset_name}.")

    print("\n" + "=" * 135)
    print("All datasets processed successfully!")


if __name__ == '__main__':
    main()
