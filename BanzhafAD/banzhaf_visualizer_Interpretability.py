import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import warnings

matplotlib.use('Agg')

from run_all_models_TSB_AD_aligned import (
    load_single_entity,
    get_dataset_entities_from_list,
    DATASET_CONFIGS,
    FILE_LIST_PATH,
    DATASET_DATA_DIR
)
from TSB_AD.models.BanzhafAD import BanzhafAD

warnings.filterwarnings("ignore")


def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_real_visualization(file_path, save_dir="./academic_plots", crop_start=45000, crop_end=50000):
    """
    Execute single physical file analysis, verify channel dimensionality,
    and save individual prototypes and time-series plots into separate PNG and PDF files[cite: 1].
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f" Loading entity: {os.path.basename(file_path)}")
    X_full, y_full, data_train = load_single_entity(file_path)

    # Use full sequence as test dataset
    data_test = X_full
    y_test = y_full

    print(" Initializing BanzhafAD engine")
    set_seed(2024)
    model = BanzhafAD(win_size=100, patch_len=10, reduce_channels=8, n_prototypes='auto')
    model.fit(data_train)
    test_scores = model.decision_function(data_test)

    # Dimensionality and active channel count (P) verification
    print("\n" + "=" * 60)
    print(" [Dimension Verification] Verifying model internal channel dimension mapping:")
    print(f"  1. Raw test data shape (X_full/data_test): {data_test.shape}")
    print(f"     -> Raw time-series channels: {data_test.shape[1]}")

    if hasattr(model, 'reduce_channels'):
        print(f"  2. Configured dimension reduction channels (reduce_channels): {model.reduce_channels}")

    if model.normal_prototypes is not None:
        protos = model.normal_prototypes.cpu().numpy()
        K, total_features = protos.shape
        # Solve quadratic equation P^2 - P - total_features = 0 to retrieve active channel count P
        P = int((1 + np.sqrt(1 + 4 * total_features)) // 2)

        print(f"  3. Learned prototype fingerprint matrix shape (normal_prototypes): {model.normal_prototypes.shape}")
        print(f"     -> Extracted number of prototypes K: {K}")
        print(f"     -> Total interaction features: {total_features}")
        print(f"  4. [Core Verification] Reconstructed active channels P via game theory: {P}")

        # Mathematical identity verification
        is_verified = (P * (P - 1) == total_features)
        print(f"     -> Math identity check: P * (P - 1) = {P} * {P - 1} = {P * (P - 1)} (Valid: {is_verified})")
        if is_verified:
            print(f"  [SUCCESS] Verification passed! Interaction matrix dimensions: {P} x {P}.")
        else:
            print("   [WARNING] Dimension check failed. Please verify diagonal removal logic.")
    else:
        print("  Warning: model.normal_prototypes is None, falling back to default channel count.")
        K = 1
        P = 10
    print("=" * 60 + "\n")
    # =========================================================================

    # Global font formatting
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['axes.unicode_minus'] = False

    # (A) Save each prototype heatmap individually
    if model.normal_prototypes is not None:
        v_max = max(abs(protos.min()), abs(protos.max()))

        for k in range(K):
            fig, ax_b = plt.subplots(figsize=(5.5, 4.8), dpi=300)

            matrix_to_show = np.full((P, P), np.nan)
            mask = np.eye(P, dtype=bool)
            matrix_to_show[~mask] = protos[k]

            sns.heatmap(
                matrix_to_show,
                ax=ax_b,
                cmap="YlGnBu",
                vmin=-v_max,
                vmax=v_max,
                mask=mask,
                cbar=True,
                cbar_kws={
                    'label': 'Interaction Intensity',
                    'shrink': 0.85,
                    'aspect': 15
                },
                square=True,
                linewidths=0.5,
                linecolor='#ffffff',
                xticklabels=False,
                yticklabels=False
            )

            # ax_b.set_ylabel("Time Patches (Temporal Nodes)", fontsize=11, fontweight='bold', labelpad=8)
            ax_b.set_yticks([0, P - 1])
            ax_b.set_yticklabels(["Patch 1", f"Patch {P}"], fontsize=9, family='serif')

            ax_b.set_xticks([0, P - 1])
            ax_b.set_xticklabels(["Patch 1", f"Patch {P}"], fontsize=9, family='serif')

            # ax_b.set_xlabel("Patch-to-Patch Interactions", fontsize=10, fontweight='bold', labelpad=6)

            for _, spine in ax_b.spines.items():
                spine.set_visible(True)
                spine.set_color('#bdc3c7')
                spine.set_linewidth(0.8)

            base_name = f"prototype_pattern_{k + 1}"
            png_path = os.path.join(save_dir, f"{base_name}.png")
            pdf_path = os.path.join(save_dir, f"{base_name}.pdf")

            plt.savefig(png_path, bbox_inches='tight', pad_inches=0.02)
            plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
            plt.close()
            print(f" Saved prototype {k + 1}: {base_name}.png / .pdf")

    # (B) Save custom window anomaly score sequence plot individually
    fig, ax_time = plt.subplots(figsize=(8.5, 3.8), dpi=300)

    total_len = len(test_scores)
    start = max(0, min(crop_start, total_len - 1))
    end = max(start + 1, min(crop_end, total_len))

    indices = np.arange(start, end)
    cropped_scores = test_scores[start:end]

    ax_time.plot(indices, cropped_scores, color='#1f77b4', linewidth=1.5, label='Anomaly Score')
    ax_time.fill_between(indices, cropped_scores, color='#1f77b4', alpha=0.08)

    if y_test is not None:
        cropped_y = y_test[start:end]
        ax_time.fill_between(
            indices, 0, max(cropped_scores) * 1.1,
            where=cropped_y > 0,
            color='#e74c3c',
            alpha=0.12,
            label='Ground Truth Anomaly',
            step='mid'
        )

    ax_time.set_ylabel("Anomaly Score", fontsize=10, fontweight='bold')
    ax_time.set_xlabel("Time Step (Sequence)", fontsize=10, fontweight='bold')
    ax_time.grid(True, linestyle=':', alpha=0.5, color='#cccccc')

    ax_time.set_xlim(start, end - 1)
    ax_time.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)

    ax_time.legend(
        loc='upper right',
        frameon=True,
        facecolor='white',
        edgecolor='none',
        fontsize=9
    )

    seq_png_path = os.path.join(save_dir, "anomaly_score_sequence.png")
    seq_pdf_path = os.path.join(save_dir, "anomaly_score_sequence.pdf")

    plt.savefig(seq_png_path, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(seq_pdf_path, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(" Saved anomaly score sequence: anomaly_score_sequence.png / .pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MSL', choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--start', type=int, default=25000, help='Custom start timestep')
    parser.add_argument('--end', type=int, default=45000, help='Custom end timestep')
    args, _ = parser.parse_known_args()

    ds_config = DATASET_CONFIGS[args.dataset]

    if args.file:
        selected_file = os.path.join(DATASET_DATA_DIR, args.file)
    else:
        file_list, _ = get_dataset_entities_from_list(FILE_LIST_PATH, DATASET_DATA_DIR, ds_config)
        selected_file = file_list[0]

    run_real_visualization(selected_file, crop_start=args.start, crop_end=args.end)


if __name__ == '__main__':
    main()
