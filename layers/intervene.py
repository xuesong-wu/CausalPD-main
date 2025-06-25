import numpy as np
import torch
import torch.nn as nn

def separate_patches(attn, threshold):
    """Separate attention values into causal and non-causal parts based on threshold.
    
    Args:
        attn: Attention map with shape [B, H, W]
        threshold: Ratio of values to be considered as causal (0~1)
    
    Returns:
        tuple: (causal_values, noncausal_values) with same shape as input
    """
    if isinstance(attn, torch.Tensor):
        B, H, W = attn.shape
        k = int(threshold * H * W)
        flattened = attn.reshape(B, -1)
        
        if k > 0:
            topk_values, topk_indices = torch.topk(flattened, k, dim=1)
            causal_mask = torch.zeros_like(flattened, dtype=torch.bool)
            causal_mask.scatter_(1, topk_indices, True)
            causal_mask = causal_mask.reshape(B, H, W)
            
            return attn * causal_mask.float(), attn * (~causal_mask).float()
        return torch.zeros_like(attn), attn

    elif isinstance(attn, np.ndarray):
        B, H, W = attn.shape
        k = int(threshold * H * W)
        flattened = attn.reshape(B, -1)
        
        if k > 0:
            topk_indices = np.argpartition(flattened, -k, axis=1)[:, -k:]
            causal_mask = np.zeros_like(flattened, dtype=bool)
            np.put_along_axis(causal_mask, topk_indices, True, axis=1)
            causal_mask = causal_mask.reshape(B, H, W)
            
            return attn * causal_mask.astype(attn.dtype), attn * (~causal_mask)
        return np.zeros_like(attn), attn

    raise TypeError("attn must be either torch.Tensor or numpy.ndarray")

def apply_intervention(z, noncausal_mask, N_b=3, decay_factor=0.5, eta=2.0, K=3):
    """Apply intervention on patches using weighted neighbor information.
    
    Args:
        z: Input tensor [bs x nvars x patch_len x patch_num]
        noncausal_mask: Non-causal mask [bs x patch_num]
        N_b: Number of neighbor patches to use
        decay_factor: Time decay factor for neighbor weights
        eta: Beta distribution parameter for mixing ratio
        K: Number of intervention samples
    
    Returns:
        Intervened tensor [(bs*K) x nvars x patch_len x patch_num]
    """
    if noncausal_mask is None:
        return z
    
    B, N, P, D = z.shape
    z_intervened = []
    
    # Calculate neighbor weights with decay
    neighbor_weights = torch.exp(-decay_factor * torch.arange(N_b, device=z.device))
    neighbor_weights = neighbor_weights / neighbor_weights.sum()
    neighbor_weights = neighbor_weights.unsqueeze(0).expand(B, -1)

    # Generate K intervention samples
    for k in range(K):
        lambdas = torch.distributions.Beta(eta * 2, eta * 2).sample((B, D)).to(z.device)
        lambdas = torch.clamp(lambdas, 0.1, 0.9)
        z_k = z.clone()

        for b in range(B):
            mask = noncausal_mask[b]
            if not mask.any():
                continue
                
            intervene_indices = torch.where(mask)[0]
            neighbor_indices = torch.arange(N_b, device=z.device).unsqueeze(0)
            start_indices = (intervene_indices - N_b + 1).clamp(min=0).unsqueeze(1)
            neighbor_indices = start_indices + neighbor_indices
            
            # Get and process neighbor patches
            neighbors = z[b, :, :, neighbor_indices.permute(1, 0)].permute(1, 2, 3, 0)
            weighted_sum = torch.matmul(neighbors, neighbor_weights[b])
            original_patches = z[b, :, :, intervene_indices]
            
            # Mix original and neighbor patches
            mix_ratios = lambdas[b, intervene_indices].view(1, 1, -1)
            mixed_patches = (1 - mix_ratios) * original_patches + mix_ratios * weighted_sum.permute(1, 2, 0)
            
            # Add small noise for diversity
            noise = torch.randn_like(mixed_patches) * (0.01 * torch.std(original_patches))
            z_k[b, :, :, intervene_indices] = mixed_patches + noise

        z_intervened.append(z_k)

    return torch.cat(z_intervened, dim=0)