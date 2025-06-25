import torch
import numpy as np
import os
from layers.intervene import separate_patches


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


def compute_attention_masks(model, setting, attention_threshold):
    attention_list = []
    with torch.no_grad():
        layer_attentions = []
        for i, layer in enumerate(model.model.backbone.encoder.layers):
            if hasattr(layer, 'attn'):
                att_avg = layer.attn.mean(dim=1)
                layer_attentions.append(att_avg.cpu().numpy())
            else:
                print(f"Layer {i} has no attribute 'attn'")
        if len(layer_attentions) > 0:
            layer_attentions = np.stack(layer_attentions, axis=0)
            mean_attention = layer_attentions.mean(axis=0)
            attention_list.append(mean_attention)

    attention_array = np.concatenate(attention_list, axis=0)
    save_path = os.path.join('./attention_maps', '/' + setting + '_att.npy')
    np.save(save_path, attention_array)
    print("Attention map saved, shape:", attention_array.shape)

    causal_mask, noncausal_mask = separate_patches(attention_array, threshold=attention_threshold)
    return causal_mask, noncausal_mask
