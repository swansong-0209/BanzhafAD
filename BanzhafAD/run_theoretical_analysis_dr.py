import os
import argparse
import inspect

from run_all_models_TSB_AD_aligned import (
    load_single_entity,
    get_dataset_entities_from_list,
    DATASET_CONFIGS,
    FILE_LIST_PATH,
    DATASET_DATA_DIR,
    Optimal_Multi_algo_HP_dict
)
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.BanzhafAD import BanzhafAD
from banzhaf_visualizer_dr import run_banzhaf_theoretical_analysis, inject_banzhaf_tracker


def main():
    print("=" * 80)
    print(" Automated Evaluation: Banzhaf Dimensionality Reduction Theoretical Verification")
    print("=" * 80)

    parser = argparse.ArgumentParser(description="BanzhafAD Dimensionality Reduction Verification")
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Target CSV file path or filename'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='SWaT',
        choices=list(DATASET_CONFIGS.keys()),
        help='Default dataset name if no specific file is provided'
    )
    args = parser.parse_args()

    target_dataset = args.dataset
    ds_config = DATASET_CONFIGS[target_dataset]

    # 1: Scan and load target entity data
    if args.file is not None:
        print(f"\n[Step 1] Parsing specified CSV file: {args.file}")
        if os.path.exists(args.file):
            selected_file = args.file
        else:
            possible_path = os.path.join(DATASET_DATA_DIR, args.file)
            if os.path.exists(possible_path):
                selected_file = possible_path
            else:
                print(f" Error: Unable to locate target file: '{args.file}'")
                return
    else:
        print(f"\n[Step 1] Automatically retrieving {target_dataset} dataset entity list")
        try:
            file_list, _ = get_dataset_entities_from_list(FILE_LIST_PATH, DATASET_DATA_DIR, ds_config)
            selected_file = file_list[0]
        except Exception as e:
            print(f" Failed to retrieve dataset files: {e}")
            return

    print(f" Selected target entity: {os.path.basename(selected_file)}")

    try:
        X_full, y_full, data_train = load_single_entity(selected_file)
        print(f" Training data loaded | Timesteps: {data_train.shape[0]} | Raw Channels: {data_train.shape[1]}")
    except Exception as e:
        print(f" Failed to load entity data: {e}")
        return

    # 2: Initialize BanzhafAD and execute training
    inject_banzhaf_tracker(BanzhafAD)
    print("\n[Step 2] Initializing BanzhafAD model")

    win_size = find_length_rank(X_full[:, 0].reshape(-1, 1), rank=1)
    print(f" ️ Adaptive sliding window size (win_size): {win_size}")

    custom_hp = {
        'win_size': win_size,
        'patch_len': 10,
        'reduce_channels': max(3, data_train.shape[1] // 2),
        'n_prototypes': 'auto',
        'num_samples': 50
    }

    if 'BanzhafAD' in Optimal_Multi_algo_HP_dict:
        official_hp = Optimal_Multi_algo_HP_dict['BanzhafAD'].copy()

        if 'window_size' in official_hp:
            official_hp['win_size'] = official_hp.pop('window_size')

        official_hp.update({
            'win_size': win_size,
            'reduce_channels': max(3, data_train.shape[1] // 2)
        })
        custom_hp = official_hp
        print(f"  [Aligned Params] Successfully sanitized and "
              f"matched the official optimal configuration from TSB_AD/HP_list.py: {custom_hp}")

    sig = inspect.signature(BanzhafAD.__init__)
    valid_params = [p.name for p in sig.parameters.values() if p.name != 'self']
    filtered_hp = {k: v for k, v in custom_hp.items() if k in valid_params}

    model = BanzhafAD(**filtered_hp)
    model.fit(data_train)
    print("   Model fitting and game-theoretic interaction matrix computation completed.")

    # 3: Theoretical analysis and visualization
    print("\n[Step 3] Running theoretical analysis engine")

    output_dir = "./academic_plots"
    run_banzhaf_theoretical_analysis(banzhaf_ad_instance=model, save_dir=output_dir)

    print(f" Plots successfully exported to: {os.path.abspath(output_dir)}/banzhaf_theoretical_verification_CR.png")
    print("=" * 80)


if __name__ == '__main__':
    main()