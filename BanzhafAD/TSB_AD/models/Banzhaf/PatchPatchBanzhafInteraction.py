import torch


class PatchPatchBanzhafInteraction:
    def __init__(self, num_samples=100, alpha=0.5):
        """
        Patch-to-Patch Banzhaf interaction computation module.
        :param num_samples: Number of Monte Carlo samples (M)
        """
        self.M = num_samples
        self.alpha = alpha

    @torch.no_grad()
    def __call__(self, patch_logits, patch_mask, patch_weight):
        """
        Compute the second-order Banzhaf interaction index for all patch pairs.

        Input dimensions:
            patch_logits: [B, P, P] -> Base association, correlation, or similarity matrix between patches
            patch_mask:   [B, P]    -> Valid mask for patches (1 for valid, 0 for masked/padded)
            patch_weight: [B, P]    -> Attention weight or importance score for each patch

        Output dimensions:
            banzhaf_matrix: [B, P, P] -> Quantified matrix of second-order interactions between patches
        """
        B, P, _ = patch_logits.size()
        device = patch_logits.device

        # 1. Construct parallel Monte Carlo random sub-coalition masks for both patch dimensions
        s_p1_base = (torch.rand(B, self.M, P, device=device) > 0.5).long()
        s_p2_base = (torch.rand(B, self.M, P, device=device) > 0.5).long()

        # 2. 4D Broadcasting: Exclude players for each specific position pair (i, j), Target shape: [B, M, P_i, P_j, P]
        s_p1 = s_p1_base.unsqueeze(2).unsqueeze(3).expand(B, self.M, P, P, P)
        s_p2 = s_p2_base.unsqueeze(2).unsqueeze(3).expand(B, self.M, P, P, P)

        diag_i = (1 - torch.eye(P, device=device)).view(1, 1, P, 1, P)
        diag_j = (1 - torch.eye(P, device=device)).view(1, 1, 1, P, P)

        s_p1 = s_p1 * diag_i
        s_p2 = s_p2 * diag_j

        # 3. Assemble the 4 coalition state masks required by the Banzhaf formula
        mask_1_4, mask_2_4 = s_p1, s_p2

        mask_1_1 = mask_1_4.clone()
        mask_2_1 = mask_2_4.clone()

        for i in range(P):
            mask_1_1[:, :, i, :, i] = 1
            mask_2_1[:, :, :, i, i] = 1

        mask_1_2, mask_2_2 = mask_1_1, mask_2_4
        mask_1_3, mask_2_3 = mask_1_4, mask_2_1

        # 4. Characteristic function computation
        def compute_v(m1, m2):
            """
            Compute spatio-temporal characteristic value based on row/column dual coalition masks.
            m1, m2 shape: [B, M, P_i, P_j, P]
            """
            # Incorporate global patch mask
            m1 = m1 * patch_mask.view(B, 1, 1, 1, P)  # [B, M, P_i, P_j, P]
            m2 = m2 * patch_mask.view(B, 1, 1, 1, P)  # [B, M, P_i, P_j, P]

            # Compute coalition size |S|
            S_size = m1.sum(dim=-1).float()  # [B, M, P_i, P_j]
            S_size_sq = torch.clamp(S_size * S_size, min=1.0)
            S_size_minus_1 = torch.clamp(S_size - 1.0, min=1.0)

            # Term 1: Info(S) -> Sum of all elements in coalition / |S|^2
            m_row = m1.unsqueeze(-1)  # [B, M, P_i, P_j, P_row, 1]
            m_col = m2.unsqueeze(-2)  # [B, M, P_i, P_j, 1, P_col]
            union_mask = m_row * m_col  # [B, M, P_i, P_j, P_row, P_col]

            # Broadcast association matrix A: [B, 1, 1, 1, P, P] -> [B, M, P_i, P_j, P, P]
            expanded_logits = patch_logits.view(B, 1, 1, 1, P, P).expand(B, self.M, P, P, P, P)
            A_S = expanded_logits * union_mask

            info_S = A_S.sum(dim=(-2, -1)) / S_size_sq  # [B, M, P_i, P_j]

            # Term 2: Temporal(S) -> Adjacent patch transitions in coalition / (|S| - 1)
            diag_adjacent = torch.diagonal(patch_logits, offset=1, dim1=1, dim2=2)
            diag_adjacent = diag_adjacent.view(B, 1, 1, 1, P - 1)

            temporal_mask = m1[..., :-1] * m2[..., 1:]  # [B, M, P_i, P_j, P - 1]

            temporal_S = torch.sum(diag_adjacent * temporal_mask, dim=-1) / S_size_minus_1  # [B, M, P_i, P_j]

            # Final value function
            v = self.alpha * info_S + (1.0 - self.alpha) * temporal_S
            return v

        # 5. Parallel second-order difference and expectation
        v1 = compute_v(mask_1_1, mask_2_1)
        v2 = compute_v(mask_1_2, mask_2_2)
        v3 = compute_v(mask_1_3, mask_2_3)
        v4 = compute_v(mask_1_4, mask_2_4)

        banzhaf_matrix = (v1 - v2 - v3 + v4).mean(dim=1)  # [B, P, P]

        # Apply global patch mask
        banzhaf_matrix = banzhaf_matrix * patch_mask.unsqueeze(2) * patch_mask.unsqueeze(1)
        return banzhaf_matrix
