import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from .Banzhaf.ChannelChannelBanzhafInteraction import ChannelChannelBanzhafInteraction
from .Banzhaf.PatchPatchBanzhafInteraction import PatchPatchBanzhafInteraction


class BanzhafAD:
    def __init__(self, win_size=100, patch_len=10, reduce_channels=None, num_samples=50,
                         n_prototypes=5, use_temporal=True, device=None):
        """
        :param win_size: Sliding window size (L)
        :param patch_len: Length of each patch
        :param reduce_channels: umber of channels to retain after reduction (None means w/o channel reduction)
        :param num_samples: Monte Carlo sample count for Banzhaf calculation (M)
        """
        self.win_size = win_size
        self.patch_len = patch_len
        self.reduce_channels = reduce_channels
        self.num_samples = num_samples
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Banzhaf interaction modules
        self.channel_game = ChannelChannelBanzhafInteraction(num_samples=self.num_samples)
        self.patch_game = PatchPatchBanzhafInteraction(num_samples=self.num_samples)

        self.scaler = StandardScaler()
        self.selected_channel_indices = None  # Retained channel indices after dimensionality reduction
        self.n_prototypes = n_prototypes
        self.use_temporal = use_temporal

        self.normal_prototypes = None  # n_prototypes

    def _to_windows(self, data):
        """Slice sequence data into sliding windows."""
        X = []
        for i in range(len(data) - self.win_size + 1):
            X.append(data[i:i + self.win_size])
        return np.array(X, dtype=np.float32)

    def _spectral_analysis(self, matrix, matrix_name="Channel Interaction", target_energy=0.90):
        """
        Perform spectral analysis on the interaction matrix to derive the dynamic Top-K value
        required to retain the target cumulative energy ratio.
        """
        # Calculate magnitude of eigenvalues in descending order
        eigenvalues = torch.linalg.eigvals(matrix).abs()
        eigenvalues, _ = torch.sort(eigenvalues, descending=True)

        total_sum = torch.sum(eigenvalues)
        C = matrix.size(0)

        if total_sum == 0:
            print(f" [Spectral Analysis Warning] The {matrix_name} "
                  f"matrix contains only zeros; spectral contributions cannot be computed.")
            return int(np.ceil(0.3 * C))

        # Calculate cumulative energy contribution
        cumulative_energy = torch.cumsum(eigenvalues, dim=0) / total_sum

        # Dynamically find minimum K required to reach target energy
        k_dynamic = torch.where(cumulative_energy >= target_energy)[0][0].item() + 1

        return k_dynamic

    def _compute_channel_reduction(self, data_train_tensor):
        """
        Step 1: Compute channel reduction via Banzhaf values and dynamic spectral thresholding.
        """
        C = data_train_tensor.size(1)

        # Normalize features and compute raw feature cross-correlations
        features_norm = torch.nn.functional.normalize(data_train_tensor.T, p=2, dim=-1)
        raw_logits = torch.mm(features_norm, features_norm.T)  # [C, C]

        # Apply softmax non-linear scaling with temperature parameter tau
        tau = 0.5
        mean_matrix = torch.softmax(raw_logits / tau, dim=-1).unsqueeze(0)  # [1, C, C]

        if torch.isnan(mean_matrix).any():
            mean_matrix = torch.nan_to_num(mean_matrix, nan=0.0)

        mask = torch.ones(1, C, device=self.device)

        # Initialize coalition weights using global variance per channel
        weight = data_train_tensor.var(dim=0).unsqueeze(0)
        weight = torch.nn.functional.softmax(weight, dim=-1)

        # Compute Banzhaf interaction base matrix for channels
        base_bi_matrix = self.channel_game(mean_matrix, mask, weight).squeeze(0)  # [C, C]

        # Normalize Banzhaf interaction matrix using Z-score standardization
        bi_std = base_bi_matrix.std() + 1e-8
        base_bi_matrix_norm = (base_bi_matrix - base_bi_matrix.mean()) / bi_std

        # Construct dynamic manifold matrix with zeroed diagonal entries
        dynamic_manifold = raw_logits.abs()
        dynamic_manifold.fill_diagonal_(0.0)

        dm_mean = dynamic_manifold.mean()
        dm_std = dynamic_manifold.std() + 1e-8
        dynamic_manifold_norm = (dynamic_manifold - dm_mean) / dm_std

        # Fuse Banzhaf interactions and raw residual dynamic manifold
        bi_matrix = base_bi_matrix_norm + dynamic_manifold_norm * 0.55

        # Compute dynamic K via spectral energy contribution
        k_dynamic = self._spectral_analysis(bi_matrix, matrix_name="Channel Banzhaf", target_energy=0.90)

        # Apply boundary safeguards for retained channel count
        k_dynamic = max(int(np.ceil(0.3 * C)), min(k_dynamic, int(np.ceil(0.9 * C))))

        # Select Top-K channels based on cumulative importance scores
        channel_importance = bi_matrix.abs().sum(dim=-1)  # [C]
        _, topk_indices = torch.topk(channel_importance, k=k_dynamic)

        self.selected_channel_indices = topk_indices.cpu().numpy()
        print(
            f"  [Banzhaf Dimensionality Reduction Result] Original channels: {C} -> "
            f"Channels retained: {k_dynamic}, indices: {self.selected_channel_indices}\n")

    def _get_temporal_sine_cosine_encoding(self, seq_len, d_model):
        """Generate positional sine-cosine temporal encodings for patches inside the window."""
        pe = torch.zeros(seq_len, d_model, device=self.device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=self.device) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _extract_patch_banzhaf(self, window_tensor):
        """
        Step 2: Partition windows into patches and compute Patch-wise Banzhaf interaction matrices.
        window_tensor: [B, win_size, C_reduced]
        """
        B, L, C = window_tensor.size()
        P = L // self.patch_len

        # Flatten patches: [B, L, C] -> [B, P, patch_len * C]
        patches = window_tensor.view(B, P, self.patch_len, C).reshape(B, P, -1)

        # Inject temporal/positional encodings if enabled
        if self.use_temporal:
            temporal_enc = self._get_temporal_sine_cosine_encoding(P, patches.size(-1))
            patches = patches + temporal_enc.unsqueeze(0)

        # Calculate cosine similarity logits between patches
        patches_norm = torch.nn.functional.normalize(patches, p=2, dim=-1)
        patch_logits = torch.bmm(patches_norm, patches_norm.transpose(1, 2))  # [B, P, P]

        patch_mask = torch.ones(B, P, device=self.device)
        patch_weight = torch.ones(B, P, device=self.device) / P

        # Compute Patch Banzhaf interaction matrices
        banzhaf_matrix = self.patch_game(patch_logits, patch_mask, patch_weight)  # [B, P, P]
        return banzhaf_matrix

    def fit(self, data_train):
        """
        Fit model on normal training data to select channels and construct normal prototypes.
        """

        self.scaler.fit(data_train)
        scaled_train = self.scaler.transform(data_train)

        train_tensor = torch.from_numpy(scaled_train).float().to(self.device)

        # 1. Channel reduction step
        if self.reduce_channels is not None and self.reduce_channels < data_train.shape[1]:
            self._compute_channel_reduction(train_tensor)
            scaled_train = scaled_train[:, self.selected_channel_indices]
        else:
            self.selected_channel_indices = np.arange(data_train.shape[1])

        # 2. Extract patches and extract Banzhaf matrices across windows
        windows = self._to_windows(scaled_train)
        if len(windows) == 0:
            return self

        windows_tensor = torch.from_numpy(windows).float().to(self.device)

        # Process in mini-batches to prevent OOM
        all_matrices = []
        batch_size = 128
        num_batches = (len(windows_tensor) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(windows_tensor), batch_size),
                      total=num_batches, desc="[BanzhafAD] Training"):
            batch_win = windows_tensor[i:i + batch_size]
            batch_bi = self._extract_patch_banzhaf(batch_win)
            all_matrices.append(batch_bi.cpu())

        all_matrices = torch.cat(all_matrices, dim=0)  # [Total_Windows, P, P]
        total_w, P, _ = all_matrices.shape

        # Mask main diagonal entries to retain off-diagonal interactions: [Total_Windows, P * (P - 1)]
        mask_off_diag = torch.ones(P, P, dtype=torch.bool)
        mask_off_diag.fill_diagonal_(False)
        flat_matrices = all_matrices[:, mask_off_diag].numpy()

        # Apply stride sampling for large sequence lengths to reduce redundancy
        stride = 5 if total_w > 20000 else 1
        train_flat_matrices = flat_matrices[::stride]

        # GMM + BIC to build prototypes
        if self.n_prototypes == 'auto':
            k_range = list(range(2, min(10, len(train_flat_matrices))))
            # k_range = [1]   # Single-Prototype Parameter Settings
            bic_scores = []
            models = {}

            for k in k_range:
                gmm = GaussianMixture(n_components=k, covariance_type='tied', random_state=42)
                gmm.fit(train_flat_matrices)
                bic_scores.append(gmm.bic(train_flat_matrices))
                models[k] = gmm

            # ====== Save Data for Visualization ======
            self.visual_k_range = k_range
            self.visual_bic_scores = bic_scores

            best_k = k_range[np.argmin(bic_scores)]
            best_gmm = models[best_k]
            print(f"  [Dynamic Prototype Selection]: Automatic decision completed! "
                  f"The optimal number of prototypes is dynamically set to: {best_k}")
        else:
            best_k = self.n_prototypes
            best_gmm = GaussianMixture(n_components=best_k, covariance_type='tied', random_state=42)
            best_gmm.fit(train_flat_matrices)
            self.visual_k_range = None
            self.visual_bic_scores = None

        # Extract prototype centers
        self.normal_prototypes = torch.from_numpy(best_gmm.means_).float().to(self.device)

        # store mask
        self.mask_off_diag = mask_off_diag.to(self.device)

        return self

    def decision_function(self, data_test):
        """
        Estimate anomaly scores for the test samples by quantifying their deviations from the game-theoretic equilibrium
        """
        scaled_test = self.scaler.transform(data_test)
        scaled_test = scaled_test[:, self.selected_channel_indices]

        windows = self._to_windows(scaled_test)
        total_len = len(data_test)

        pointwise_error = np.zeros(total_len)
        counts = np.zeros(total_len)

        global_test_proto_tensors = []

        windows_tensor = torch.from_numpy(windows).float().to(self.device)
        batch_size = 128
        num_batches = (len(windows_tensor) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(0, len(windows_tensor), batch_size),
                          total=num_batches, desc="[BanzhafAD] Inference"):
                batch_win = windows_tensor[i:i + batch_size]

                # Compute the patch-level Banzhaf game matrix within the current batch window
                batch_bi = self._extract_patch_banzhaf(batch_win)  # [B_batch, P, P]

                B_curr = batch_bi.size(0)

                # Extract off-diagonal interaction features
                batch_bi_flat = batch_bi[:, self.mask_off_diag]

                # Compute distance residual to normal prototype set
                bi_expanded = batch_bi_flat.unsqueeze(1)
                proto_expanded = self.normal_prototypes.unsqueeze(0)
                delta_all = bi_expanded - proto_expanded  # [B_curr, n_prototypes, P * (P - 1)]


                dist_to_protos = delta_all.pow(2).mean(dim=-1)  # [B_curr, best_k]

                # Select closest matching prototype index
                best_proto_indices = torch.argmin(dist_to_protos, dim=-1)  # [B_curr]
                global_test_proto_tensors.append(best_proto_indices)

                # Get residual magnitudes relative to closest prototype
                delta_flat = delta_all[torch.arange(B_curr), best_proto_indices].abs() # delta_flat: [B_curr, P * (P - 1)]

                # Aggregate Top-K largest residuals (Top 20%)
                topk_num = max(1, int(0.20 * delta_flat.size(1)))
                topk_values, _ = torch.topk(delta_flat, k=topk_num, dim=-1)  # [B_curr, topk_num]

                physical_error = torch.mean(topk_values ** 2, dim=-1)
                error_np = physical_error.cpu().numpy()

                # Map window-level errors back to pointwise sequence indices
                for b in range(len(error_np)):
                    global_start = i + b
                    global_end = global_start + self.win_size
                    pointwise_error[global_start:global_end] += error_np[b]
                    counts[global_start:global_end] += 1.0

        if global_test_proto_tensors:
            self.visual_test_state_chain = torch.cat(global_test_proto_tensors, dim=0).cpu().numpy()
        else:
            self.visual_test_state_chain = np.array([])

        counts[counts == 0] = 1.0
        scores = pointwise_error / counts
        return scores
