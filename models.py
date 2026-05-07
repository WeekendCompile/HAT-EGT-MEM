import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 750, scale_factor: float = 1.0):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size * scale_factor)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class DualScaleTemporalEncoder(nn.Module):
    """
    Unified temporal encoder that captures both fine-grained local motion 
    and long-range temporal dependencies without relying on sequence-length branching.
    """
    def __init__(self, embedding_dim, num_heads, dropout):
        super(DualScaleTemporalEncoder, self).__init__()
        
        # Local scale: Depthwise 1D Convolutions for short-term temporal dynamics
        self.local_encoder = nn.Conv1d(
            in_channels=embedding_dim, 
            out_channels=embedding_dim, 
            kernel_size=5, 
            padding=2, 
            groups=embedding_dim # Depthwise convolution for efficiency
        )
        self.local_norm = nn.LayerNorm(embedding_dim)
        
        # Global scale: Transformer Self-Attention for long-range dependencies
        self.global_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            activation='gelu',
            batch_first=False
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x shape: [seq_len, batch, dim]
        """
        seq_len, batch, dim = x.shape
        
        # Local processing
        local_x = x.permute(1, 2, 0)  # [batch, dim, seq_len]
        local_x = self.local_encoder(local_x)
        local_x = local_x.permute(2, 0, 1)  # [seq_len, batch, dim]
        local_x = self.local_norm(local_x + x)  # Residual connection
        
        # Global processing
        global_x = self.global_encoder(local_x)
        
        # Fusion
        fused = self.fusion(torch.cat([local_x, global_x], dim=-1))
        
        return fused

class MemoryUnit(nn.Module):
    """
    Retentive Adaptive Gated Memory (RAGM).

    A differentiable content-addressable memory bank treated as an LRU-by-utility
    cache. Frames query the bank, an importance gate decides whether to write,
    a per-slot persistence score governs eviction (low-utility slots get
    overwritten first), and a soft erase vector controls *what* gets thrown out
    inside a slot. Persistence is updated via EMA of read attention mass, so
    frequently-retrieved concepts are carried forward.

    Causal/online: the bank is initialized fresh per sample and updated step by
    step over the temporal axis using only past frames.

    Inputs:
        x: [seq_len, batch, dim]
    Outputs:
        x_aug:    [seq_len, batch, dim]   memory-augmented frame stream
        proto:    [K, batch, dim]         final prototype slots (for aux head)
    """
    def __init__(self, embedding_dim, num_slots=16, momentum=0.9, dropout=0.3):
        super(MemoryUnit, self).__init__()
        self.K = num_slots
        self.d = embedding_dim
        self.momentum = momentum

        self.slot_init = nn.Parameter(torch.zeros(num_slots, embedding_dim))
        nn.init.normal_(self.slot_init, std=0.02)

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.scale = embedding_dim ** -0.5

        self.write_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        self.erase_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        self.write_content = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )

        self.fuse = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )
        self.out_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        seq_len, batch, dim = x.shape

        M = self.slot_init.unsqueeze(0).expand(batch, -1, -1).contiguous()
        persistence = torch.zeros(batch, self.K, device=x.device)

        outs = []
        for t in range(seq_len):
            x_t = x[t]
            q_t = self.q_proj(x_t)
            k_t = self.k_proj(M)
            v_t = self.v_proj(M)

            attn_logits = torch.einsum('bd,bkd->bk', q_t, k_t) * self.scale
            attn = F.softmax(attn_logits, dim=-1)
            r_t = torch.einsum('bk,bkd->bd', attn, v_t)

            persistence = self.momentum * persistence + (1.0 - self.momentum) * attn

            outs.append(self.fuse(torch.cat([x_t, r_t], dim=-1)))

            novelty = 1.0 - attn.max(dim=-1, keepdim=True).values
            g_w = self.write_gate(torch.cat([x_t, r_t], dim=-1)) * novelty

            evict_logits = -persistence * 4.0
            evict = F.softmax(evict_logits, dim=-1)

            xt_exp = x_t.unsqueeze(1).expand(-1, self.K, -1)
            slot_query = torch.cat([xt_exp, M], dim=-1)
            erase = self.erase_gate(slot_query)
            write = self.write_content(slot_query)

            alpha = (g_w * evict).unsqueeze(-1)
            M = M * (1.0 - alpha * erase) + alpha * write

            persistence = persistence + evict * g_w * 0.1

        x_aug = self.out_norm(torch.stack(outs, dim=0) + x)
        proto = M.transpose(0, 1).contiguous()
        return x_aug, proto


class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature = opt["feat_dim"]
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        n_enc_layer = opt["enc_layer"]
        n_enc_head = opt["enc_head"]
        n_dec_layer = opt["dec_layer"]
        n_dec_head = opt["dec_head"]
        n_seglen = opt["segment_size"]
        self.anchors = opt["anchors"]
        self.anchors_stride = []
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0
        self.use_memory = bool(opt.get("use_memory", False))
        self.num_memory_slots = int(opt.get("num_memory_slots", 16))
        
        # Enhanced feature reduction with dynamic dropout
        self.feature_reduction_rgb = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2),
            nn.Dropout(dropout * 0.5)
        )
        self.feature_reduction_flow = nn.Sequential(
            nn.Linear(self.n_feature//2, n_embedding_dim//2),
            nn.GELU(),
            nn.LayerNorm(n_embedding_dim//2),
            nn.Dropout(dropout * 0.5)
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            n_embedding_dim, 
            dropout=dropout * 0.5,
            maxlen=400,
            scale_factor=0.5
        )
        
        # Optional Memory Unit (pre-DSTE) — toggleable via opt['use_memory']
        if self.use_memory:
            self.memory_unit = MemoryUnit(
                embedding_dim=n_embedding_dim,
                num_slots=self.num_memory_slots,
                momentum=0.9,
                dropout=dropout
            )
            self.snip_head = nn.Sequential(
                nn.Linear(n_embedding_dim, n_embedding_dim // 2),
                nn.GELU(),
                nn.LayerNorm(n_embedding_dim // 2),
                nn.Dropout(dropout),
            )
            self.snip_classifier = nn.Linear(
                self.num_memory_slots * (n_embedding_dim // 2), n_class
            )

        # Unified Dual-Scale Temporal Encoder
        self.temporal_encoder = DualScaleTemporalEncoder(
            embedding_dim=n_embedding_dim,
            num_heads=n_enc_head,
            dropout=dropout
        )
        
        # Main encoder (adaptive layers)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_enc_head, 
                dropout=dropout * (0.5 if i < 2 else 1.0),  # Lower dropout for initial layers
                activation='gelu'
            ) for i in range(n_enc_layer)
        ])
        self.encoder_norm = nn.LayerNorm(n_embedding_dim)
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_dec_head, 
                dropout=dropout, 
                activation='gelu'
            ), 
            n_dec_layer, 
            nn.LayerNorm(n_embedding_dim)
        )
        

        
        # Enhanced classification and regression heads
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.GELU(), 
            nn.LayerNorm(n_embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, n_class)
        )
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), 
            nn.GELU(), 
            nn.LayerNorm(n_embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, 2)
        )
        
        self.decoder_token = nn.Parameter(torch.Tensor(len(self.anchors), 1, n_embedding_dim))
        nn.init.normal_(self.decoder_token, std=0.01)
        
        # Additional normalization layers
        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # Enhanced feature processing
        inputs = inputs.float()
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
        base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize
        seq_len = base_x.shape[0]
        
        # Apply positional encoding
        pe_x = self.positional_encoding(base_x)

        snip_cls = None
        if self.use_memory:
            pe_x, proto = self.memory_unit(pe_x)
            proto_feat = self.snip_head(proto)
            proto_feat = proto_feat.permute(1, 0, 2).reshape(proto_feat.shape[1], -1)
            snip_cls = self.snip_classifier(proto_feat)

        # Unified Dual-Scale Temporal Processing
        temporal_features = self.temporal_encoder(pe_x)
        
        # Standard encoder processing
        encoded_x = temporal_features
        for layer in self.encoder_layers:
            encoded_x = layer(encoded_x)
        
        # Apply encoder normalization
        encoded_x = self.encoder_norm(encoded_x)
        encoded_x = self.norm1(encoded_x)
        
        # Decoder processing
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)
        decoded_x = self.decoder(decoder_token, encoded_x)
        
        # Add residual connection and normalization
        decoded_x = self.norm2(decoded_x + self.dropout1(decoder_token))
        
        decoded_x = decoded_x.permute([1, 0, 2])
        
        anc_cls = self.classifier(decoded_x)
        anc_reg = self.regressor(decoded_x)

        if self.use_memory:
            return anc_cls, anc_reg, snip_cls
        return anc_cls, anc_reg


class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class=opt["num_of_class"]-1
        n_seglen=opt["segment_size"]
        n_embedding_dim=2*n_seglen
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        # FC layers for the 2 streams
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        #inputs - batch x seq_len x class
        
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x
