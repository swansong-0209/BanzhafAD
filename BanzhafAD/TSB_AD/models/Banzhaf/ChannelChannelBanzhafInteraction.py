import torch


class ChannelChannelBanzhafInteraction:
    def __init__(self, num_samples=100, alpha=0.5):
        """
        Channel-to-Channel Banzhaf interaction computation module.
        :param num_samples: Number of Monte Carlo samples (M)
        :param alpha: Weight coefficient balancing information preservation and interaction energy
        """
        self.M = num_samples
        self.alpha = alpha

    @torch.no_grad()
    def __call__(self, channel_logits, channel_mask, channel_weight=None):
        """
        Compute the second-order Banzhaf interaction index for all channel pairs.

        Input dimensions:
            channel_logits: [B, C, C] -> Association matrix or self-attention matrix between channels
            channel_mask:   [B, C]    -> Valid mask for channels (used to filter out dropped sensors)
            channel_weight: [B, C]    -> Baseline weights assigned to each channel

        Output dimensions:
            banzhaf_matrix: [B, C, C] -> Quantified matrix of second-order joint contributions
        """
        B, C, _ = channel_logits.size()
        device = channel_logits.device

        # 1. Construct parallel Monte Carlo random sub-coalition masks
        s_c1_base = (torch.rand(B, self.M, C, device=device) > 0.5).long()
        s_c2_base = (torch.rand(B, self.M, C, device=device) > 0.5).long()

        # 2. 5D explicit expansion mechanism, Target: [B, M, C_i, C_j, C_k]
        s_c1 = s_c1_base.unsqueeze(2).unsqueeze(3).expand(B, self.M, C, C, C)
        s_c2 = s_c2_base.unsqueeze(2).unsqueeze(3).expand(B, self.M, C, C, C)

        diag_i = 1 - torch.eye(C, device=device).view(1, 1, C, 1, C)
        diag_j = 1 - torch.eye(C, device=device).view(1, 1, 1, C, C)

        s_c1 = s_c1 * diag_i
        s_c2 = s_c2 * diag_j

        # 3. Assemble the 4 required coalition states
        mask_1_4, mask_2_4 = s_c1, s_c2

        mask_1_1 = mask_1_4.clone()
        mask_2_1 = mask_2_4.clone()

        for i in range(C):
            mask_1_1[:, :, i, :, i] = 1
            mask_2_1[:, :, :, i, i] = 1

        mask_1_2, mask_2_2 = mask_1_1, mask_2_4
        mask_1_3, mask_2_3 = mask_1_4, mask_2_1

        # 4. Compute characteristic functions
        def compute_v(m1, m2):
            # Incorporate global sensor clipping mask
            m1 = m1 * channel_mask.view(B, 1, 1, 1, C)   # [B, M, C_i, C_j, C_k]
            m2 = m2 * channel_mask.view(B, 1, 1, 1, C)   # [B, M, C_i, C_j, C_k]

            m_row = m1.unsqueeze(-1)  # [B, M, C_i, C_j, C_row, 1]
            m_col = m2.unsqueeze(-2)  # [B, M, C_i, C_j, 1, C_col]
            union_mask = m_row * m_col  # [B, M, C_i, C_j, C_row, C_col]

            # Compute coalition size |S|
            S_size = m1.sum(dim=-1).float()  # [B, M, C_i, C_j]
            S_size_sq = torch.clamp(S_size * S_size, min=1.0)

            # Broadcast association matrix A: [B, 1, 1, 1, C, C] -> [B, M, C_i, C_j, C, C]
            expanded_logits = channel_logits.view(B, 1, 1, 1, C, C).expand(B, self.M, C, C, C, C)

            # Filter association sub-matrix A_S within the coalition
            A_S = expanded_logits * union_mask

            # Information Preservation: Sum of elements in coalition / |S|^2
            info_preservation = A_S.sum(dim=(-2, -1)) / S_size_sq  # [B, M, C_i, C_j]

            # Interaction Energy: Squared Frobenius norm / |S|^2
            interaction_energy = torch.sum(A_S ** 2, dim=(-2, -1)) / S_size_sq  # [B, M, C_i, C_j]

            # Final value function
            v = self.alpha * info_preservation + (1.0 - self.alpha) * interaction_energy
            return v

        # 5. Parallel second-order difference and expectation
        v1 = compute_v(mask_1_1, mask_2_1)
        v2 = compute_v(mask_1_2, mask_2_2)
        v3 = compute_v(mask_1_3, mask_2_3)
        v4 = compute_v(mask_1_4, mask_2_4)

        banzhaf_matrix = (v1 - v2 - v3 + v4).mean(dim=1)  # [B, C, C]

        # Apply global sensor exclusion mask
        banzhaf_matrix = banzhaf_matrix * channel_mask.unsqueeze(2) * channel_mask.unsqueeze(1)
        return banzhaf_matrix
