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


class AdaptiveSequenceProcessor(nn.Module):
    """
    Unified sequence processor optimized for both short and long sequences.
    """
    def __init__(self, embedding_dim, num_heads, dropout, max_span=200, min_span=8):
        super(AdaptiveSequenceProcessor, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.max_span = max_span
        self.min_span = min_span
        
        # Sequence length thresholds
        self.short_threshold = 16
        self.long_threshold = 64
        
        # Short sequence processing (enhanced for better feature capture)
        self.short_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=min(num_heads, 8),  # Increased heads for short sequences
            dropout=dropout * 0.5,  # Reduced dropout
            batch_first=False
        )
        
        # Lightweight encoder for short sequences
        self.short_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=min(num_heads, 8),
            dropout=dropout * 0.5,
            activation='gelu',
            batch_first=False
        )
        
        # Long sequence processing (dual-range attention)
        self.local_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads // 2,
            dropout=dropout,
            batch_first=False
        )
        self.global_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads // 2,
            dropout=dropout,
            batch_first=False
        )
        
        # Adaptive span prediction
        self.span_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 3),  # short, local, global
            nn.Softmax(dim=-1)
        )
        
        # Relevance scoring for long sequences
        self.relevance_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temporal decay parameters
        self.temporal_decay_local = nn.Parameter(torch.tensor(0.9))
        self.temporal_decay_global = nn.Parameter(torch.tensor(0.8))
        
        # Feature fusion with enhanced flexibility
        self.feature_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )
        
        # Adaptive gating
        self.adaptive_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, memory_bank=None):
        """
        Adaptive forward pass that switches between short and long processing.
        """
        seq_len, batch_size, dim = features.shape
        
        if seq_len <= self.short_threshold:
            return self._process_short_sequence(features)
        elif seq_len <= self.long_threshold:
            return self._process_medium_sequence(features)
        else:
            return self._process_long_sequence(features, memory_bank)
    
    def _process_short_sequence(self, features):
        """Enhanced processing for short sequences."""
        seq_len, batch_size, dim = features.shape
        
        # Enhanced self-attention
        attended_features, attention_weights = self.short_attention(
            features, features, features
        )
        
        # Lightweight encoder pass
        encoded_features = self.short_encoder(attended_features)
        
        # Apply gating
        gate_weights = self.adaptive_gate(encoded_features)
        output = encoded_features * gate_weights
        
        # Residual connection and normalization
        output = self.layer_norm(output + features)
        
        return output, attention_weights.mean(dim=1), []
    
    def _process_medium_sequence(self, features):
        """Balanced processing for medium sequences."""
        seq_len, batch_size, dim = features.shape
        
        # Compute context for span prediction
        context = torch.mean(features, dim=0)
        span_weights = self.span_predictor(context)
        
        # Local attention on recent portion
        local_span = min(self.min_span * 2, features.shape[0])
        local_features = features[-local_span:]
        local_attended, local_attn = self.local_attention(
            local_features, local_features, local_features
        )
        
        # Global attention on subsampled sequence
        if features.shape[0] > 16:
            step = max(1, features.shape[0] // 16)
            global_features = features[::step]
            global_attended, global_attn = self.global_attention(
                global_features, global_features, global_features
            )
            
            # Upsample global features
            global_upsampled = global_attended.repeat_interleave(step, dim=0)
            if global_upsampled.shape[0] > features.shape[0]:
                global_upsampled = global_upsampled[:features.shape[0]]
            elif global_upsampled.shape[0] < features.shape[0]:
                padding = features.shape[0] - global_upsampled.shape[0]
                global_upsampled = torch.cat([
                    global_upsampled,
                    global_upsampled[-padding:]
                ], dim=0)
        else:
            global_upsampled = local_attended
        
        # Extend local features to match sequence length
        local_extended = torch.cat([
            features[:-local_span],
            local_attended
        ], dim=0)
        
        # Weighted combination with span weights
        combined = torch.cat([local_extended * span_weights[:, 1:2], 
                            global_upsampled * span_weights[:, 2:3]], dim=-1)
        fused_features = self.feature_fusion(combined)
        
        # Apply adaptive gating
        gate_weights = self.adaptive_gate(fused_features)
        output = fused_features * gate_weights
        
        # Residual connection and normalization
        output = self.layer_norm(output + features)
        
        attention_weights = torch.ones(seq_len, batch_size, device=features.device)
        return output, attention_weights, [local_span]
    
    def _process_long_sequence(self, features, memory_bank=None):
        """Comprehensive processing for long sequences."""
        seq_len, batch_size, dim = features.shape
        
        # Compute adaptive spans
        context = torch.mean(features, dim=0)
        span_weights = self.span_predictor(context)
        
        # Process each batch separately for efficiency
        local_features = []
        global_features = []
        
        for b in range(batch_size):
            # Local processing
            local_span = min(self.min_span * 4, seq_len)
            local_history = features[-local_span:, b:b+1, :]
            
            if local_history.shape[0] > 0:
                local_context = context[b:b+1, :].unsqueeze(0).expand(local_history.shape[0], -1, -1)
                local_combined = torch.cat([local_history, local_context], dim=-1)
                local_scores = self.relevance_scorer(local_combined).squeeze(-1).squeeze(-1)
                
                # Apply temporal decay
                local_positions = torch.arange(local_history.shape[0], device=features.device, dtype=torch.float)
                local_decay = self.temporal_decay_local ** (local_history.shape[0] - 1 - local_positions)
                local_weights = F.softmax(local_scores + torch.log(local_decay + 1e-8), dim=0)
                
                local_feat = (local_history.squeeze(1) * local_weights.unsqueeze(-1)).sum(dim=0)
            else:
                local_feat = torch.zeros(self.embedding_dim, device=features.device)
            
            # Global processing (subsampled)
            if seq_len > 32:
                step = max(1, seq_len // 32)
                global_history = features[::step, b:b+1, :]
            else:
                global_history = features[:, b:b+1, :]
            
            if global_history.shape[0] > 0:
                global_context = context[b:b+1, :].unsqueeze(0).expand(global_history.shape[0], -1, -1)
                global_combined = torch.cat([global_history, global_context], dim=-1)
                global_scores = self.relevance_scorer(global_combined).squeeze(-1).squeeze(-1)
                
                global_positions = torch.arange(global_history.shape[0], device=features.device, dtype=torch.float)
                global_decay = self.temporal_decay_global ** (global_history.shape[0] - 1 - global_positions)
                global_weights = F.softmax(global_scores + torch.log(global_decay + 1e-8), dim=0)
                
                global_feat = (global_history.squeeze(1) * global_weights.unsqueeze(-1)).sum(dim=0)
            else:
                global_feat = torch.zeros(self.embedding_dim, device=features.device)
            
            local_features.append(local_feat)
            global_features.append(global_feat)
        
        # Combine features
        local_stack = torch.stack(local_features, dim=0)
        global_stack = torch.stack(global_features, dim=0)
        
        # Weighted combination based on predicted spans
        combined = torch.cat([
            local_stack * span_weights[:, 1:2],  # local weight
            global_stack * span_weights[:, 2:3]  # global weight
        ], dim=-1)
        
        # Fusion and expansion to sequence length
        fused = self.feature_fusion(combined)
        fused_expanded = fused.unsqueeze(0).expand(seq_len, -1, -1)
        
        # Apply adaptive gating
        gate_weights = self.adaptive_gate(fused_expanded)
        output = fused_expanded * gate_weights
        
        # Residual connection and normalization
        output = self.layer_norm(output + features)
        
        attention_weights = torch.ones(seq_len, batch_size, device=features.device) / seq_len
        actual_spans = [local_span, min(self.max_span, seq_len)]
        
        return output, attention_weights, actual_spans


class HierarchicalContextEncoder(nn.Module):
    """
    Adaptive hierarchical encoder optimized for varying sequence lengths.
    """
    def __init__(self, embedding_dim, num_heads, dropout):
        super(HierarchicalContextEncoder, self).__init__()
        
        # Multi-scale encoders
        self.fine_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dropout=dropout * 0.5,  # Reduced for short sequences
            activation='gelu',
            batch_first=False
        )
        
        self.coarse_encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=max(1, num_heads // 2), 
            dropout=dropout, 
            activation='gelu',
            batch_first=False
        )
        
        # Adaptive scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim)
        )
        
        # Dynamic feature weighting for short sequences
        self.short_weighting = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temporal consistency
        self.temporal_consistency = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        
        # Adaptive feature selection
        self.feature_selector = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 2),
            nn.Softmax(dim=-1)
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, sequence_length_ratio=1.0):
        """
        Adaptive forward pass based on sequence characteristics.
        """
        seq_len, batch_size, dim = features.shape
        
        # Fine-grained processing
        fine_features = self.fine_encoder(features)
        
        # Enhanced weighting for short sequences
        if seq_len <= 16:
            weights = self.short_weighting(fine_features)
            fine_features = fine_features * weights
            coarse_upsampled = fine_features
        else:
            # Coarse-grained processing (adaptive based on sequence length)
            downsample_factor = min(4, max(2, seq_len // 16))
            coarse_features = features.permute(1, 2, 0)  # [batch, dim, seq]
            coarse_features = F.avg_pool1d(
                coarse_features, 
                kernel_size=downsample_factor, 
                stride=downsample_factor, 
                padding=0
            )
            coarse_features = coarse_features.permute(2, 0, 1)  # [seq//factor, batch, dim]
            
            # Encode coarse features
            coarse_encoded = self.coarse_encoder(coarse_features)
            
            # Upsample back to original length
            coarse_upsampled = coarse_encoded.permute(1, 2, 0)  # [batch, dim, seq//factor]
            coarse_upsampled = F.interpolate(
                coarse_upsampled, 
                size=seq_len, 
                mode='linear', 
                align_corners=False
            )
            coarse_upsampled = coarse_upsampled.permute(2, 0, 1)  # [seq, batch, dim]
        
        # Adaptive feature selection
        combined_for_selection = torch.cat([fine_features, coarse_upsampled], dim=-1)
        selection_weights = self.feature_selector(combined_for_selection)
        
        # Weighted combination (bias toward fine features for short sequences)
        selection_weights = selection_weights * torch.tensor([1.5 if seq_len <= 16 else 1.0, 
                                                            0.5 if seq_len <= 16 else 1.0], 
                                                            device=features.device).view(1, 1, 2)
        selected_features = (fine_features * selection_weights[:, :, 0:1] + 
                           coarse_upsampled * selection_weights[:, :, 1:2])
        
        # Temporal consistency for longer sequences
        if seq_len > 16:
            consistency_features = selected_features.permute(1, 2, 0)  # [batch, dim, seq]
            consistency_features = self.temporal_consistency(consistency_features)
            consistency_features = consistency_features.permute(2, 0, 1)  # [seq, batch, dim]
            
            # Fusion
            fused_features = self.scale_fusion(torch.cat([selected_features, consistency_features], dim=-1))
        else:
            fused_features = self.scale_fusion(torch.cat([selected_features, fine_features], dim=-1))
        
        # Final normalization
        output = self.layer_norm(fused_features)
        output = self.dropout(output)
        
        return output


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
        
        # Sequence length thresholds for adaptive processing
        self.short_threshold = 16
        self.medium_threshold = 64
        self.long_threshold = 128
        
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
        
        # Adaptive positional encoding
        self.positional_encoding = PositionalEncoding(
            n_embedding_dim, 
            dropout=dropout * 0.5,  # Reduced for short sequences
            maxlen=400,
            scale_factor=0.5  # Lower frequency for short sequences
        )
        
        # Unified adaptive sequence processor
        self.adaptive_processor = AdaptiveSequenceProcessor(
            embedding_dim=n_embedding_dim,
            num_heads=n_enc_head,
            dropout=dropout,
            max_span=200,
            min_span=8
        )
        
        # Hierarchical context encoder
        self.hierarchical_encoder = HierarchicalContextEncoder(
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
        
        # Context integration
        self.context_integration = nn.Sequential(
            nn.Linear(n_embedding_dim * 2, n_embedding_dim),
            nn.GELU(),
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
        
        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))
        
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
        
        # Determine sequence complexity ratio
        sequence_length_ratio = min(1.0, seq_len / self.long_threshold)
        
        # Adaptive processing based on sequence length
        if seq_len <= self.short_threshold:
            # Short sequence: Enhanced processing
            processed_features, _, _ = self.adaptive_processor(pe_x)
            
            # Use only initial encoder layers for short sequences
            encoded_x = processed_features
            for i in range(min(3, len(self.encoder_layers))):  # Increased to 3 layers
                encoded_x = self.encoder_layers[i](encoded_x)
            
        elif seq_len <= self.medium_threshold:
            # Medium sequence: Balanced processing
            processed_features, _, _ = self.adaptive_processor(pe_x)
            hierarchical_features = self.hierarchical_encoder(processed_features, sequence_length_ratio)
            
            # Integrate adaptive and hierarchical features
            integrated_features = self.context_integration(
                torch.cat([processed_features, hierarchical_features], dim=-1)
            )
            
            # Standard encoder processing
            encoded_x = integrated_features
            for layer in self.encoder_layers:
                encoded_x = layer(encoded_x)
            
        else:
            # Long sequence: Full hierarchical processing
            hierarchical_features = self.hierarchical_encoder(pe_x, sequence_length_ratio)
            processed_features, _, _ = self.adaptive_processor(hierarchical_features)
            
            # Integrate features
            integrated_features = self.context_integration(
                torch.cat([hierarchical_features, processed_features], dim=-1)
            )
            
            # Full encoder processing
            encoded_x = integrated_features
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

