__all__ = ['CausalPD_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.CausalPD_layers import *
from layers.RevIN import RevIN
from layers.intervene import apply_intervention

# Cell
class CausalPD_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, meta_dim:int, stride:int, max_seq_len:Optional[int]=1024,
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, use_intervention:bool=False, N_b:int=3, decay_factor:float=0.5, K:int=3, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1

        # Intervention parameters
        self.use_intervention = use_intervention
        self.N_b = N_b
        self.decay_factor = decay_factor
        self.K = K
        self.sample_weights = nn.Parameter(torch.ones(K) / K) 

        # Backbone
        self.backbone = CausalPDEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len, meta_dim = meta_dim,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z, ext, meta, noncausal_mask=None):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
            ext = self.padding_patch_layer(ext)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        ext = ext.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        ext = ext.permute(0, 1, 3, 2)

        # Apply intervention if needed
        if self.use_intervention:
            z = apply_intervention(z, noncausal_mask, self.N_b, self.decay_factor, K=self.K)  # z: [(bs*K) x nvars x patch_len x patch_num]
            ext = ext.repeat(self.K, 1, 1, 1)
            if meta is not None:
                meta = meta.repeat(self.K, 1)

        # model
        z = self.backbone(z, ext, meta)                                                                # z: [(bs*K) x nvars x d_model x patch_num]
        
        if self.use_intervention:
            z = z.view(self.K, -1, self.n_vars, z.size(2), z.size(3))
            z = self.head(z)  # z: [K x bs x nvars x target_window]
        else:
            z = self.head(z)  # z: [bs x nvars x target_window]
        
        # denorm
        if self.revin: 
            if self.use_intervention:
                z = z.permute(0, 1, 3, 2)  # [K x bs x target_window x nvars]
                z = self.revin_layer(z, 'denorm')
                z = z.permute(0, 1, 3, 2)  # [K x bs x nvars x target_window]
            else:
                z = z.permute(0, 2, 1)
                z = self.revin_layer(z, 'denorm')
                z = z.permute(0, 2, 1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
        
class FeatureWiseDynamicModulation(nn.Module):
    def __init__(self, meta_dim: int, d_model: int):
        super().__init__()
        # Static metadata processing
        self.embed_meta = nn.Linear(meta_dim, d_model)        # Project static features to d_model
        self.norm_meta = nn.LayerNorm(d_model)                # Normalize static features
        self.proj_meta = nn.Linear(d_model, d_model)          # Project to final embedding

        # Exogenous variables processing
        self.norm_ext = nn.LayerNorm(d_model)                 
        self.proj_ext = nn.Linear(d_model, d_model)           

        # Feature fusion
        self.fuse = nn.Linear(2 * d_model, d_model)           
        
        # Modulation gates
        self.global_gate = nn.Parameter(torch.ones(1, 1, 1, d_model))  
        self.dynamic_gate = nn.Linear(d_model, d_model)                
        self.final_norm = nn.LayerNorm(d_model)                        

    def forward(self, x: Tensor, ext: Tensor, meta: Tensor) -> Tensor:
        B, N, P, D = x.shape
        
        # Process static metadata
        gamma = self.embed_meta(meta)                      # [bs, node num, d_model]
        gamma = self.norm_meta(gamma)                          # [bs, node num, d_model]
        gamma = self.proj_meta(gamma)                          # [bs, node num, d_model]
        gamma = gamma.unsqueeze(2).expand(-1, -1, P, -1)  # [bs, node num, patch num, d_model]
        
        # Process exogenous variables
        beta = self.norm_ext(ext)                         
        beta = self.proj_ext(beta)                           
        beta = beta.mean(dim=1, keepdim=True).expand(-1, N, -1, -1)  # [bs, node num, patch num, d_model]
        
        # Fuse features
        ctx = self.fuse(torch.cat([gamma, beta], dim=-1))  
        
        # Compute modulation weights
        dynamic_weight = torch.sigmoid(self.dynamic_gate(ctx))  
        
        # Apply modulation
        global_modulated = x * self.global_gate        
        dynamic_modulated = x * dynamic_weight         
        
        # Combine and normalize
        output = global_modulated + dynamic_modulated 
        output = self.final_norm(output) 

        return output, ctx# [bs, node num, patch num, d_model]
    
class CausalPDEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024, meta_dim = 0,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.W_E = nn.Linear(patch_len, d_model)
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.fdm = FeatureWiseDynamicModulation(meta_dim=meta_dim, d_model=d_model)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = ETEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x, ext, meta) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.transpose(2, 3).contiguous()  # [B, node_num, patch_num, patch_len]
        ext = ext.transpose(2, 3).contiguous()  # [B, variable_num, patch_num, patch_len]

        #
        x = self.W_P(x)                               # x: [bs x nvars x patch_num x d_model]
        ext = self.W_E(ext)                              # [bs, variable num, patch num, d_model]

        fused, ctx = self.fdm(x, ext, meta)

        B, N, P, D = fused.shape
        u = fused.view(B * N, P, D)  # [B * node_num, patch_num, d_model]
        u = self.dropout(u + self.W_pos)  # [bs * node num, patch num, d_model]

        # Encoder
        z = self.encoder(u, ctx=ctx)                                                      # z: [bs * nvars x patch_num x d_model]
        z = z.view(B, N, P, D).transpose(2, 3).contiguous()  # [B, node_num, d_model, patch_num]                                                 # z: [bs x nvars x d_model x patch_num]
        
        return z    
            
            
    
# Cell
class ETEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([ETEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, ctx:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, ctx=ctx, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, ctx=ctx, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class ETEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=True,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, ctx:Optional[Tensor]=None, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, ctx=ctx, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, ctx=ctx, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, ctx:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # If ctx is provided, reshape it to multi-head form
        if ctx is not None:
            ctx = ctx.view(bs, -1, self.n_heads, self.d_k).transpose(1,2)  # [bs x n_heads x seq_len x d_k]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, ctx=ctx, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, ctx=ctx, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x seq_len x d_v], attn: [bs x n_heads x seq_len x seq_len], scores: [bs x n_heads x seq_len x seq_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x seq_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        # Add bilinear tensor parameter with dimension [d_k x d_k], where d_k is the dimension of each head
        self.bilinear = nn.Parameter(torch.randn(head_dim, head_dim))
        # Add learnable scaling factor to control bias influence
        self.bias_scale = nn.Parameter(torch.tensor(0.1))
        # Add bias normalization layer
        self.bias_norm = nn.LayerNorm(head_dim)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, ctx:Tensor=None, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            ctx            : [bs x n_heads x max_q_len x d_k]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # Calculate bilinear bias
        if ctx is not None:
            # Normalize ctx
            ctx = self.bias_norm(ctx)  # [bs x n_heads x seq_len x d_k]
            
            # Calculate bilinear transformation
            ctx_bilinear = torch.matmul(ctx, self.bilinear)  # [bs x n_heads x seq_len x d_k]
            bilinear_bias = torch.matmul(ctx_bilinear, ctx.transpose(-2, -1))  # [bs x n_heads x seq_len x seq_len]
            bilinear_bias = F.normalize(bilinear_bias, dim=-1)
            bilinear_bias = bilinear_bias * self.bias_scale
            
            # Add to attention weights
            attn_weights = attn_weights + bilinear_bias
            attn_weights = F.normalize(attn_weights, dim=-1)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

