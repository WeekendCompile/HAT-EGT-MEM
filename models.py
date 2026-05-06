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


# ---------------------------------------------------------------------------
# Module 1: Dual-Scale Temporal Encoder
# ---------------------------------------------------------------------------
# Captures both fine-grained local motion patterns (via depthwise 1D
# convolutions) and long-range temporal dependencies (via self-attention)
# through a unified local-global feature extraction mechanism.
# Operates on the FULL temporal sequence before any split.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Module 2: Long-Range Memory Unit (LMU)
# ---------------------------------------------------------------------------
# Introduces a targeted mechanism for context retention and retrieval.
# After multi-scale encoding, the long-range context portion of the
# temporal sequence is compressed into a compact set of learnable memory
# tokens via cross-attention. A content-aware gate selectively weights
# task-relevant information. An auxiliary classification head ensures
# the memory tokens encode meaningful action semantics via direct
# gradient supervision.
#
# Unlike naive approaches that stack redundant layers or increase
# parameter count, the LMU serves a clear functional role: preserving
# and structuring long-horizon context to guide current-window predictions.
# ---------------------------------------------------------------------------
class LongRangeMemoryUnit(nn.Module):
    """
    Compresses long-range encoded context into compact memory tokens via
    cross-attention, applies content-aware gating, and produces auxiliary
    classification for direct gradient supervision.
    """
    def __init__(self, embedding_dim, num_classes, n_memory_tokens=8,
                 n_heads=4, n_layers=2, dropout=0.3):
        super(LongRangeMemoryUnit, self).__init__()
        self.n_memory_tokens = n_memory_tokens
        
        # Learnable memory query tokens
        self.memory_tokens = nn.Parameter(
            torch.zeros(n_memory_tokens, 1, embedding_dim))
        nn.init.normal_(self.memory_tokens, std=0.02)
        
        # Cross-attention compressor: memory tokens attend to context features
        self.memory_compressor = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embedding_dim, nhead=n_heads,
                dropout=dropout, activation='gelu'),
            num_layers=n_layers,
            norm=nn.LayerNorm(embedding_dim))
        
        # Content-aware gate conditioned on context summary
        self.memory_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.GELU(),
            nn.Linear(embedding_dim // 4, embedding_dim),
            nn.Sigmoid())
        
        # Auxiliary classification head for direct gradient supervision
        self.memory_head = nn.Sequential(
            nn.Linear(n_memory_tokens * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes))

    def forward(self, context_x):
        """
        context_x: [context_len, batch, dim] — long-range encoded features
        Returns:
            memory_out: [n_memory_tokens, batch, dim] — compressed memory
            memory_cls: [batch, num_classes] — auxiliary classification
        """
        batch = context_x.shape[1]
        
        # Cross-attention compression: memory tokens query the context
        mem_tokens = self.memory_tokens.expand(-1, batch, -1)
        memory_out = self.memory_compressor(mem_tokens, context_x)
        
        # Content-aware gating: selectively weight memory dimensions
        context_summary = context_x.mean(dim=0)           # [batch, dim]
        gate = self.memory_gate(context_summary)           # [batch, dim]
        memory_out = memory_out * gate.unsqueeze(0)        # [n_tokens, batch, dim]
        
        # Auxiliary classification on compressed memory
        mem_flat = memory_out.permute(1, 0, 2).reshape(batch, -1)
        memory_cls = self.memory_head(mem_flat)             # [batch, num_classes]
        
        return memory_out, memory_cls


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
        self.short_window_size = 16  # Current prediction window
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0
        
        # ---- Stage 0: Modality-Specific Feature Reduction ----
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
        
        # ---- Stage 1: Full-Sequence Temporal Encoding ----
        # Positional encoding for the full temporal sequence
        self.positional_encoding = PositionalEncoding(
            n_embedding_dim, 
            dropout=dropout * 0.5,
            maxlen=400,
            scale_factor=0.5
        )
        
        # Dual-Scale Temporal Encoder: local conv + global attention
        self.temporal_encoder = DualScaleTemporalEncoder(
            embedding_dim=n_embedding_dim,
            num_heads=n_enc_head,
            dropout=dropout
        )
        
        # Standard encoder stack with progressive dropout
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embedding_dim, 
                nhead=n_enc_head, 
                dropout=dropout * (0.5 if i < 2 else 1.0),
                activation='gelu'
            ) for i in range(n_enc_layer)
        ])
        self.encoder_norm = nn.LayerNorm(n_embedding_dim)
        
        # ---- Stage 2: Long-Range Memory Unit ----
        # Compresses context region into compact memory tokens
        self.memory_unit = LongRangeMemoryUnit(
            embedding_dim=n_embedding_dim,
            num_classes=n_class,
            n_memory_tokens=8,
            n_heads=n_dec_head,
            n_layers=2,
            dropout=dropout
        )
        
        # ---- Stage 3: Anchor Decoder (full sequence + memory) ----
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
        
        self.decoder_token = nn.Parameter(torch.Tensor(len(self.anchors), 1, n_embedding_dim))
        nn.init.normal_(self.decoder_token, std=0.01)
        
        self.decoder_norm = nn.LayerNorm(n_embedding_dim)
        self.decoder_dropout = nn.Dropout(dropout)
        
        # ---- Stage 4: Prediction Heads ----
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

    def forward(self, inputs):
        # ---- Stage 0: Modality-specific feature reduction ----
        inputs = inputs.float()
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2])
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:])
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
        base_x = base_x.permute([1,0,2])  # [seq_len, batch, dim]
        
        # ---- Stage 1: Full-sequence temporal encoding ----
        # All frames benefit from multi-scale encoding before any split
        pe_x = self.positional_encoding(base_x)
        temporal_features = self.temporal_encoder(pe_x)
        
        encoded_x = temporal_features
        for layer in self.encoder_layers:
            encoded_x = layer(encoded_x)
        encoded_x = self.encoder_norm(encoded_x)
        
        # ---- Stage 2: Long-Range Memory Unit ----
        # Compress the long-range context portion into compact memory tokens.
        # The context region (first 48 frames) is further from the prediction
        # point and benefits most from structured compression.
        context_x = encoded_x[:-self.short_window_size]   # [48, B, D]
        memory_out, memory_cls = self.memory_unit(context_x)
        
        # ---- Stage 3: Memory-augmented anchor decoding ----
        # The decoder attends to the FULL encoded sequence PLUS the
        # compressed memory tokens. This preserves all original temporal
        # information while providing structured long-range summaries
        # as additional keys/values for the anchor queries.
        decoder_memory = torch.cat([encoded_x, memory_out], dim=0)  # [64+8, B, D]
        
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)
        decoded_x = self.decoder(decoder_token, decoder_memory)
        decoded_x = self.decoder_norm(decoded_x + self.decoder_dropout(decoder_token))
        
        decoded_x = decoded_x.permute([1, 0, 2])  # [B, n_anchors, D]
        
        # ---- Stage 4: Prediction ----
        anc_cls = self.classifier(decoded_x)
        anc_reg = self.regressor(decoded_x)
        
        return anc_cls, anc_reg, memory_cls


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
