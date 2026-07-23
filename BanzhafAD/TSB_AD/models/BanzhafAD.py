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
        :param reduce_channels: Number of channels to retain after dimensionality reduction (None for no reduction)
        :param num_samples: Number of Banzhaf Monte Carlo samples (M)
        """
        self.win_size = win_size
        self.patch_len = patch_len
        self.reduce_channels = reduce_channels
        self.num_samples = num_samples
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # parallel Banzhaf interaction modules
        self.channel_game = ChannelChannelBanzhafInteraction(num_samples=self.num_samples)
        self.patch_game = PatchPatchBanzhafInteraction(num_samples=self.num_samples)

        self.scaler = StandardScaler()
        self.selected_channel_indices = None  # Selected channel indices after dimensionality reduction
        self.n_prototypes = n_prototypes
        self.use_temporal = use_temporal

        self.normal_prototypes = None  # multiple prototype

    def _to_windows(self, data):
        """Slice sequence into sliding windows"""
        X = []
        for i in range(len(data) - self.win_size + 1):
            X.append(data[i:i + self.win_size])
        return np.array(X, dtype=np.float32)

    def _spectral_analysis(self, matrix, matrix_name="Channel Interaction", target_energy=0.90):
        """
        Perform multi-ratio spectral analysis on the interaction matrix,
        print cumulative energy contribution, and return Top-K needed to reach target energy.
        """
        # Compute magnitude of eigenvalues and sort in descending order
        eigenvalues = torch.linalg.eigvals(matrix).abs()
        eigenvalues, _ = torch.sort(eigenvalues, descending=True)

        total_sum = torch.sum(eigenvalues)
        C = matrix.size(0)

        if total_sum == 0:
            print(
                f" [Spectral Analysis Warning] Matrix {matrix_name} is all zeros, cannot calculate spectral contribution.")
            return int(np.ceil(0.3 * C))  # Return default fallback value

        # 2. Compute cumulative energy contribution curve
        cumulative_energy = torch.cumsum(eigenvalues, dim=0) / total_sum

        # Dynamically search for minimum features K required to reach >= 90% cumulative energy
        k_dynamic = torch.where(cumulative_energy >= target_energy)[0][0].item() + 1

        return k_dynamic

    def _compute_channel_reduction(self, data_train_tensor):
        """
        Step 1: Perform dimension selection using inter-channel Banzhaf index,
        incorporating spectral analysis validation and adaptive dimension reduction.
        """
        C = data_train_tensor.size(1)

        # Utilize inter-channel dynamic feature dot products and magnify non-diagonal micro-interactions using temperature coefficient
        features_norm = torch.nn.functional.normalize(data_train_tensor.T, p=2, dim=-1)
        raw_logits = torch.mm(features_norm, features_norm.T)  # [C, C]

        # 2. Suppress diagonal dominance through non-linear scaling and transfer absolute values to game matrix
        tau = 0.5
        mean_matrix = torch.softmax(raw_logits / tau, dim=-1).unsqueeze(0)  # [1, C, C]

        if torch.isnan(mean_matrix).any():
            mean_matrix = torch.nan_to_num(mean_matrix, nan=0.0)

        mask = torch.ones(1, C, device=self.device)
        weight = data_train_tensor.var(dim=0).unsqueeze(0)
        weight = torch.nn.functional.softmax(weight, dim=-1)

        # 3. Calculate underlying Banzhaf backbone between channels
        base_bi_matrix = self.channel_game(mean_matrix, mask, weight).squeeze(0)  # [C, C]

        # Normalize base_bi_matrix to eliminate magnitude collaps
        bi_std = base_bi_matrix.std() + 1e-8
        base_bi_matrix_norm = (base_bi_matrix - base_bi_matrix.mean()) / bi_std

        # Residual manifold reshapingt
        dynamic_manifold = raw_logits.abs()
        dynamic_manifold.fill_diagonal_(0.0)  # Remove diagonal dominance

        # Normalize dynamic_manifold and merge
        dm_mean = dynamic_manifold.mean()
        dm_std = dynamic_manifold.std() + 1e-8
        dynamic_manifold_norm = (dynamic_manifold - dm_mean) / dm_std

        # Blend normalized components
        bi_matrix = base_bi_matrix_norm + dynamic_manifold_norm * 0.55

        # Invoke spectral analysis function
        k_dynamic = self._spectral_analysis(bi_matrix, matrix_name="Channel Banzhaf", target_energy=0.90)

        # Apply boundary protection constraints
        k_dynamic = max(int(np.ceil(0.3 * C)), min(k_dynamic, int(np.ceil(0.9 * C))))

        # Extract Top-K channels
        channel_importance = bi_matrix.abs().sum(dim=-1)  # [C]
        _, topk_indices = torch.topk(channel_importance, k=k_dynamic)

        self.selected_channel_indices = topk_indices.cpu().numpy()
        print(f"  [Banzhaf Reduction Summary] Original Channels: {C} -> Retained: {k_dynamic}, Selected Indices: {self.selected_channel_indices}\n")

    def _get_temporal_sine_cosine_encoding(self, seq_len, d_model):
        """Generate sine-cosine positional/temporal encodings for patches inside the window"""
        pe = torch.zeros(seq_len, d_model, device=self.device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=self.device) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _extract_patch_banzhaf(self, window_tensor):
        """
        Step 2: Partition patches for single/batch windows and calculate patch Banzhaf interaction matrix
        window_tensor: [B, win_size, C_reduced]
        """
        B, L, C = window_tensor.size()
        P = L // self.patch_len  # number of patches

        # Partition sequence into patches and flatten features: [B, L, C] -> [B, P, patch_len * C]
        patches = window_tensor.view(B, P, self.patch_len, C).reshape(B, P, -1)

        # Inject explicit temporal positional encodings
        if self.use_temporal:
            temporal_enc = self._get_temporal_sine_cosine_encoding(P, patches.size(-1))
            patches = patches + temporal_enc.unsqueeze(0)

        # Compute patch similarity/correlation matrix as baseline game response score
        # Measure patch logits using normalized dot product
        patches_norm = torch.nn.functional.normalize(patches, p=2, dim=-1)
        patch_logits = torch.bmm(patches_norm, patches_norm.transpose(1, 2))  # [B, P, P]

        patch_mask = torch.ones(B, P, device=self.device)
        patch_weight = torch.ones(B, P, device=self.device) / P

        # Invoke fully parallel Patch Banzhaf interaction module
        banzhaf_matrix = self.patch_game(patch_logits, patch_mask, patch_weight)  # [B, P, P]
        return banzhaf_matrix

    def fit(self, data_train):
        """
        Learn channel reduction rules and normal patch game baselines
        """
        self.scaler.fit(data_train)
        scaled_train = self.scaler.transform(data_train)

        train_tensor = torch.from_numpy(scaled_train).float().to(self.device)

        # 1. Channel dimension reduction
        if self.reduce_channels is not None and self.reduce_channels < data_train.shape[1]:
            self._compute_channel_reduction(train_tensor)
            scaled_train = scaled_train[:, self.selected_channel_indices]
        else:
            self.selected_channel_indices = np.arange(data_train.shape[1])

        # 2. Partition patches and establish baseline game patterns for normal states
        windows = self._to_windows(scaled_train)
        if len(windows) == 0:
            return self

        # Sample normal windows to calculate average interaction baseline
        windows_tensor = torch.from_numpy(windows).float().to(self.device)

        all_matrices = []
        batch_size = 128
        num_batches = (len(windows_tensor) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(windows_tensor), batch_size),
                      total=num_batches, desc="[BanzhafAD] Training"):
            batch_win = windows_tensor[i:i + batch_size]
            batch_bi = self._extract_patch_banzhaf(batch_win)
            all_matrices.append(batch_bi.cpu())

        # Concatenate and cluster into game prototypes via GMM
        all_matrices = torch.cat(all_matrices, dim=0)  # [Total_Windows, P, P]
        total_w, P, _ = all_matrices.shape

        # Strip main diagonal using Tensor mask
        mask_off_diag = torch.ones(P, P, dtype=torch.bool)
        mask_off_diag.fill_diagonal_(False)

        # Extract non-diagonal elements across all windows in a single step using broadcasting
        # Extracted shape: [Total_Windows, P * (P - 1)]
        flat_matrices = all_matrices[:, mask_off_diag].numpy()

        # Equal-interval stride sampling in time domain to eliminate high-frequency temporal redundancy
        stride = 5 if total_w > 20000 else 1
        train_flat_matrices = flat_matrices[::stride]

        # Dynamic adaptive optimal prototype search via GMM + BIC
        if self.n_prototypes == 'auto':
            print("  [Dynamic Prototype Selection] Searching for optimal prototype count via GMM + BIC")
            k_range = list(range(2, min(10, len(train_flat_matrices))))
            bic_scores = []
            models = {}

            for k in k_range:
                gmm = GaussianMixture(n_components=k, covariance_type='tied', random_state=42)
                gmm.fit(train_flat_matrices)
                bic_scores.append(gmm.bic(train_flat_matrices))
                models[k] = gmm

            # Store data for subsequent visualization
            self.visual_k_range = k_range
            self.visual_bic_scores = bic_scores

            best_k = k_range[np.argmin(bic_scores)]
            best_gmm = models[best_k]
            print(f"  [Dynamic Prototype Selection] Automatic decision complete! Optimal prototype count set to: {best_k}")
        else:
            best_k = self.n_prototypes
            best_gmm = GaussianMixture(n_components=best_k, covariance_type='tied', random_state=42)
            best_gmm.fit(train_flat_matrices)
            self.visual_k_range = None
            self.visual_bic_scores = None

        # Extract prototypes from GMM cluster centers
        self.normal_prototypes = torch.from_numpy(best_gmm.means_).float().to(self.device)
        self.mask_off_diag = mask_off_diag.to(self.device)

        return self

    def decision_function(self, data_test):
        """
        Test Inference: Calculate real-time deviation from Patch game equilibrium as anomaly scores
        """
        scaled_test = self.scaler.transform(data_test)
        # Apply dimension reduction rules determined during training
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
                # Calculate Patch Banzhaf game matrix for current window batch
                batch_bi = self._extract_patch_banzhaf(batch_win)  # [B_batch, P, P]

                # Top-K Anomaly Residual Aggregation
                B_curr = batch_bi.size(0)

                # xtract non-diagonal interaction features for current batch -> [B_curr, P * (P - 1)]
                batch_bi_flat = batch_bi[:, self.mask_off_diag]

                # Expand dimensions and compute residuals against all normal prototypes
                # batch_bi_flat: [B_curr, 1, P*(P-1)]
                # normal_prototypes: [1, best_k, P*(P-1)]
                bi_expanded = batch_bi_flat.unsqueeze(1)
                proto_expanded = self.normal_prototypes.unsqueeze(0)
                delta_all = bi_expanded - proto_expanded  # [B_curr, n_prototypes, P * (P - 1)]

                # Calculate game residual distances in off-diagonal space
                dist_to_protos = delta_all.pow(2).mean(dim=-1)  # [B_curr, best_k]

                # Find best matching prototype index
                best_proto_indices = torch.argmin(dist_to_protos, dim=-1)  # [B_curr]
                global_test_proto_tensors.append(best_proto_indices)

                # Extract actual residual matrix delta_flat under best matching prototype: [B_curr, P * (P - 1)]
                delta_flat = delta_all[torch.arange(B_curr), best_proto_indices].abs()

                # Extract Top 20% most severely disrupted local game interaction pairs (Top-K)
                topk_num = max(1, int(0.20 * delta_flat.size(1)))
                topk_values, _ = torch.topk(delta_flat, k=topk_num, dim=-1)  # [B_curr, topk_num]

                # Residual score
                physical_error = torch.mean(topk_values ** 2, dim=-1)
                error_np = physical_error.cpu().numpy()


                # Smoothly project window scores back onto time steps across time series
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
