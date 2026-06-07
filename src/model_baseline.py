"""
Cross-Domain HAR Baseline Model — Frozen LLM Backbone (Phase 1).

Architecture:
  SensorInstanceNorm -> SensorEmbedding (Conv1D patching, d_model native)
  -> LearnablePositionalEncoding
  -> Frozen LLM Backbone (truncated, LayerNorm trainable)
  -> ActivityProjection (last-token classification head)

Used for the multi-LLM baseline study comparing 7 LLM backbones on
cross-domain HAR across 4 datasets (UCI, Shoaib, MotionSense, HHAR).
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# ============================================================================
# SENSOR INSTANCE NORMALIZATION
# ============================================================================

class SensorInstanceNorm(nn.Module):
    """Per-channel per-sample normalization. (N, T, C) -> (N, T, C)."""
    def __init__(self, num_channels=6):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_channels, affine=True)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

# ============================================================================
# SENSOR EMBEDDING (SensorInd + SensorSeg + Conv1D)
# ============================================================================

class SensorEmbedding(nn.Module):
    """Convert IMU windows to token embeddings via patching + Conv1D projection.

    (N, 120, 6) -> patch into (N, C*m, patch_len) -> Conv1D -> (N, num_tokens, d_model)
    Total tokens = num_channels * segment_count = 6 * 8 = 48
    """
    def __init__(self, num_channels=6, segment_count=8, patch_length=15, d_model=768):
        super().__init__()
        self.num_channels = num_channels
        self.segment_count = segment_count
        self.patch_length = patch_length
        self.d_model = d_model
        self.num_tokens = num_channels * segment_count

        # Non-overlapping Conv1D: kernel = stride = patch_length
        self.projection = nn.Conv1d(1, d_model, kernel_size=patch_length,
                                     stride=patch_length, bias=True)

    def forward(self, x):
        N = x.shape[0]
        x = x.transpose(1, 2)  # (N, 6, 120)
        x = x.reshape(N, self.num_channels, self.segment_count, self.patch_length)
        x = x.reshape(N * self.num_channels * self.segment_count, 1, self.patch_length)
        x = self.projection(x)  # (N*48, d_model, 1)
        x = x.squeeze(-1).reshape(N, self.num_tokens, self.d_model)
        return x

# ============================================================================
# LEARNABLE POSITIONAL ENCODING
# ============================================================================

class LearnablePositionalEncoding(nn.Module):
    """Learnable position vectors added to token embeddings."""
    def __init__(self, num_tokens=48, d_model=768):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x):
        return x + self.pe

# ============================================================================
# LLM BACKBONE — Native AutoModel + inputs_embeds
# ============================================================================

class LLMBackbone(nn.Module):
    """Pretrained LLM decoder with in-place layer truncation + selective freezing.

    Uses AutoModel.forward(inputs_embeds=x) so that RoPE, causal masks, and
    all architecture-specific logic are handled natively by HuggingFace.

    Keeps: first N transformer blocks (proportional to total depth)
    Freezes: attention + feedforward weights
    Trains: layer normalization parameters
    """
    def __init__(self, llm_name='gpt2', layers_to_keep=4, d_model=768,
                 freeze_attention=True, freeze_feedforward=True, train_layernorm=True):
        super().__init__()

        config = AutoConfig.from_pretrained(llm_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            llm_name, config=config,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )

        # In-place layer truncation
        self._truncate(layers_to_keep)

        self.actual_d_model = config.hidden_size

        # Dimension projections (only when d_model != hidden_size)
        self.input_proj = nn.Linear(d_model, self.actual_d_model) if d_model != self.actual_d_model else None
        self.output_proj = nn.Linear(self.actual_d_model, d_model) if d_model != self.actual_d_model else None

        # Selective freezing
        self._freeze(freeze_attention, freeze_feedforward, train_layernorm)

    def _truncate(self, n):
        """In-place truncation to first n transformer layers."""
        if hasattr(self.model, 'h'):                    # GPT-2
            self.model.h = self.model.h[:n]
        elif hasattr(self.model, 'layers'):             # Llama, Qwen, Gemma, Phi, Pythia
            self.model.layers = self.model.layers[:n]
        elif hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'layers'):  # OPT
            self.model.decoder.layers = self.model.decoder.layers[:n]
        else:
            raise ValueError(f"Cannot find layer list to truncate in {type(self.model)}")

    def _freeze(self, freeze_attn, freeze_ff, train_ln):
        # Freeze everything first
        for p in self.model.parameters():
            p.requires_grad = False

        # Selectively unfreeze based on parameter name patterns
        LN_KEYS = ['ln_', 'layernorm', 'layer_norm', 'norm', 'input_layernorm',
                    'post_attention_layernorm', 'final_layernorm']
        ATTN_KEYS = ['attn', 'attention', 'self_attn', 'q_proj', 'k_proj',
                     'v_proj', 'o_proj', 'c_attn', 'c_proj']
        FF_KEYS = ['mlp', 'feedforward', 'fc1', 'fc2', 'c_fc', 'gate_proj',
                   'up_proj', 'down_proj', 'dense']

        for name, param in self.model.named_parameters():
            nl = name.lower()
            is_ln = any(k in nl for k in LN_KEYS)
            is_attn = any(k in nl for k in ATTN_KEYS)
            is_ff = any(k in nl for k in FF_KEYS)

            if is_ln and train_ln:
                param.requires_grad = True
            elif is_attn and not freeze_attn:
                param.requires_grad = True
            elif is_ff and not freeze_ff:
                param.requires_grad = True

    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)
        # Native forward handles RoPE, causal masks, etc. automatically
        out = self.model(inputs_embeds=x)
        x = out.last_hidden_state
        if self.output_proj is not None:
            x = self.output_proj(x)
        return x

# ============================================================================
# ACTIVITY PROJECTION (Classification Head)
# ============================================================================

class ActivityProjection(nn.Module):
    """FeatureMixing classification head. Uses last token (causal attention)."""
    def __init__(self, d_model=768, num_classes=4, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        return self.head(x[:, -1, :])  # Last token

# ============================================================================
# FULL LLM4HAR MODEL
# ============================================================================

class LLM4HAR(nn.Module):
    """LLM4HAR: Cross-domain HAR via pretrained LLM decoder blocks.

    Args:
        num_channels: IMU channels (default 6: acc_xyz + gyro_xyz)
        window_samples: Samples per window (default 120 = 6s at 20Hz)
        segment_count: Patches per channel (default 8)
        d_model: Embedding dimension (default 768 for GPT-2)
        num_classes: Activity classes (default 4)
        llm_name: HuggingFace model ID
        llm_layers_to_keep: Decoder layers to retain
        freeze_attention: Freeze self-attention weights
        freeze_feedforward: Freeze feedforward weights
        train_layernorm: Train layer normalization
        dropout: Dropout rate for classification head
    """
    def __init__(self, num_channels=6, window_samples=120, segment_count=8,
                 d_model=768, num_classes=4, llm_name='gpt2', llm_layers_to_keep=4,
                 freeze_attention=True, freeze_feedforward=True, train_layernorm=True,
                 dropout=0.1):
        super().__init__()
        patch_length = window_samples // segment_count  # 15
        num_tokens = num_channels * segment_count       # 48

        self.instance_norm = SensorInstanceNorm(num_channels)
        self.sensor_embedding = SensorEmbedding(num_channels, segment_count,
                                                 patch_length, d_model)
        self.positional_encoding = LearnablePositionalEncoding(num_tokens, d_model)
        self.llm_backbone = LLMBackbone(llm_name, llm_layers_to_keep, d_model,
                                         freeze_attention, freeze_feedforward,
                                         train_layernorm)
        self.activity_projection = ActivityProjection(d_model, num_classes, dropout)

    def forward(self, x):
        """x: (batch, 120, 6) -> logits: (batch, num_classes)"""
        x = self.instance_norm(x)              # (N, 120, 6)
        x = self.sensor_embedding(x)           # (N, 48, d_model)
        x = self.positional_encoding(x)        # (N, 48, d_model)
        x = self.llm_backbone(x)               # (N, 48, d_model)
        return self.activity_projection(x)     # (N, num_classes)

    def get_param_stats(self):
        """Return trainable vs frozen parameter counts per component."""
        stats = {}
        total_t, total_f = 0, 0
        for name, mod in [('instance_norm', self.instance_norm),
                          ('sensor_embedding', self.sensor_embedding),
                          ('positional_encoding', self.positional_encoding),
                          ('llm_backbone', self.llm_backbone),
                          ('activity_projection', self.activity_projection)]:
            t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            f = sum(p.numel() for p in mod.parameters() if not p.requires_grad)
            stats[name] = {'trainable': t, 'frozen': f}
            total_t += t; total_f += f
        stats['total_trainable'] = total_t
        stats['total_frozen'] = total_f
        stats['total_params'] = total_t + total_f
        return stats