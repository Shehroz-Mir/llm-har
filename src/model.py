"""
Cross-Domain HAR Model — Frozen LLM Backbone with Sensor Adaptation.

Architecture:
  SensorInstanceNorm -> SensorEmbedding (position-augmented Conv1D patching)
  -> LearnablePositionalEncoding
  -> [ReprogrammingLayer: gated cross-attention with learned prototypes]
  -> Frozen LLM Backbone (truncated, LayerNorm trainable)
  -> ActivityProjection (last-token classification head)
  -> [ReconstructionHead: auxiliary reconstruction loss]

Components:
  - SensorEmbedding: concatenates time indices with patch values before Conv1D
  - ReprogrammingLayer: gated cross-attention mapping sensor patches to LLM space
  - AlignmentHead: next-patch prediction with MSE + FFT loss (Stage 1 training)
  - ReconstructionHead: sensor reconstruction auxiliary loss (Stage 2 training)
  - LLMBackbone: AutoModel with in-place layer truncation and selective freezing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# LayerNorm parameter name patterns
LN_KEYS = ['ln_', 'layernorm', 'layer_norm', 'input_layernorm',
           'post_attention_layernorm', 'final_layernorm', 'norm']

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
# SENSOR EMBEDDING (Position-Augmented Patching + Conv1D)
# ============================================================================

class SensorEmbedding(nn.Module):
    """Convert IMU windows to token embeddings via position-augmented patching.
    Each patch concatenates sensor values with normalized time indices (TimeSense).
    Conv1D kernel_size = 2 * patch_length to cover (value, time_index) pairs.
    (N, 120, 6) -> (N, 48, d_model)
    """
    def __init__(self, num_channels=6, segment_count=8, patch_length=15, d_model=768):
        super().__init__()
        self.num_channels = num_channels
        self.segment_count = segment_count
        self.patch_length = patch_length
        self.d_model = d_model
        self.num_tokens = num_channels * segment_count

        # Kernel covers (value, time_index) pairs per patch: 2 * patch_length
        input_dim = 2 * patch_length
        self.projection = nn.Conv1d(1, d_model, kernel_size=input_dim,
                                     stride=input_dim, bias=True)

    def forward(self, x):
        N, T, C = x.shape  # (N, 120, 6)
        # Create normalized time indices [0, 1/(T-1), ..., 1]
        t_idx = torch.linspace(0, 1, T, device=x.device)           # (T,)
        t_idx = t_idx.unsqueeze(0).unsqueeze(-1).expand(N, T, C)   # (N, T, C)

        # Transpose to (N, C, T) for both values and time indices
        x = x.transpose(1, 2)                                       # (N, 6, 120)
        t_idx = t_idx.transpose(1, 2)                                # (N, 6, 120)

        # Reshape into patches: (N, 6, 8, 15)
        x = x.reshape(N, C, self.segment_count, self.patch_length)
        t_idx = t_idx.reshape(N, C, self.segment_count, self.patch_length)

        # Concatenate values with time indices within each patch: (N, 6, 8, 30)
        x = torch.cat([x, t_idx], dim=-1)

        # Flatten for Conv1d: (N*48, 1, 30)
        x = x.reshape(N * self.num_tokens, 1, 2 * self.patch_length)
        x = self.projection(x)                                      # (N*48, d_model, 1)
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
# REPROGRAMMING LAYER (Time-LLM + OpenTSLM Gated Cross-Attention)
# ============================================================================

class ReprogrammingLayer(nn.Module):
    """Maps sensor patch embeddings into LLM-friendly representations
    via gated cross-attention with learned prototypes.

    Learns a small set of 'activity prototypes' that bridge the sensor
    modality to the LLM's embedding space. Cross-attention lets each
    sensor patch select relevant prototype information.
    Gated residual (OpenTSLM): gate starts at 0 (identity pass-through),
    model gradually learns to incorporate prototypes.
    """
    def __init__(self, d_model, n_prototypes=64, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, d_model) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        # Learnable gate (OpenTSLM): starts at 0 for stable early training
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (N, 48, d_model)
        N = x.size(0)
        proto = self.prototypes.unsqueeze(0).expand(N, -1, -1)  # (N, n_proto, d_model)
        # Patches attend to prototypes: Q=patches, K=V=prototypes
        reprogrammed, _ = self.cross_attn(x, proto, proto)
        return self.norm(x + self.gate * reprogrammed)  # Gated residual

# ============================================================================
# ALIGNMENT HEAD (LLM4TS / SensorLLM + TimeSense FFT Loss)
# ============================================================================

class AlignmentHead(nn.Module):
    """Predicts next patch embedding (autoregressive alignment).
    Uses combined MSE + FFT frequency-domain loss (TimeSense) to capture
    periodic patterns critical for HAR.
    Used only during Stage 1 self-supervised training.
    """
    def __init__(self, d_model):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, llm_output, target_embeddings):
        """Compute next-patch prediction loss (MSE + FFT).
        Args:
            llm_output: (N, 48, d_model) — LLM hidden states
            target_embeddings: (N, 48, d_model) — sensor embeddings (pre-LLM)
        Returns:
            Combined MSE + FFT loss (scalar)
        """
        pred = self.predictor(llm_output[:, :-1, :])    # positions 0..46
        target = target_embeddings[:, 1:, :].detach()   # positions 1..47
        # Time-domain MSE loss
        mse_loss = F.mse_loss(pred, target)
        # Frequency-domain FFT loss (TimeSense) — captures periodic patterns
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        fft_loss = F.mse_loss(pred_fft.abs(), target_fft.abs())
        return mse_loss + fft_loss

# ============================================================================
# RECONSTRUCTION HEAD (TimeSense Auxiliary Loss)
# ============================================================================

class ReconstructionHead(nn.Module):
    """Reconstructs sensor embeddings from LLM output (TimeSense auxiliary loss).
    Encourages the LLM to preserve sensor information during classification.
    Uses combined MSE + FFT loss for reconstruction quality.
    """
    def __init__(self, d_model):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, llm_output, target_embeddings):
        """Compute reconstruction loss.
        Args:
            llm_output: (N, 48, d_model) — LLM hidden states
            target_embeddings: (N, 48, d_model) — sensor embeddings (pre-LLM)
        Returns:
            Combined MSE + FFT reconstruction loss (scalar)
        """
        recon = self.decoder(llm_output)
        target = target_embeddings.detach()
        # Time-domain MSE
        mse_loss = F.mse_loss(recon, target)
        # Frequency-domain FFT loss
        recon_fft = torch.fft.rfft(recon, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        fft_loss = F.mse_loss(recon_fft.abs(), target_fft.abs())
        return mse_loss + fft_loss

# ============================================================================
# LLM BACKBONE — Native AutoModel + inputs_embeds
# ============================================================================

class LLMBackbone(nn.Module):
    """Pretrained LLM decoder with in-place layer truncation and selective freezing.
    Supports GPT-2 wpe zeroing to fix double positional encoding.
    """
    def __init__(self, llm_name='gpt2', layers_to_keep=4,
                 freeze_attention=True, freeze_feedforward=True, train_layernorm=True,
                 config=None):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(llm_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            llm_name, config=config,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )

        # Fix 1: Disable GPT-2 internal wpe to prevent double positional encoding
        if llm_name == 'gpt2':
            self.model.wpe.weight.data.zero_()
            self.model.wpe.weight.requires_grad = False

        # In-place layer truncation
        self._truncate(layers_to_keep)

        # Selective freezing
        self._freeze(freeze_attention, freeze_feedforward, train_layernorm)

    def _truncate(self, n):
        """In-place truncation to first n transformer layers."""
        if hasattr(self.model, 'h'):                    # GPT-2
            self.model.h = self.model.h[:n]
        elif hasattr(self.model, 'layers'):             # Llama, Qwen, Gemma
            self.model.layers = self.model.layers[:n]
        elif hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'layers'):
            self.model.decoder.layers = self.model.decoder.layers[:n]
        else:
            raise ValueError(f"Cannot find layer list to truncate in {type(self.model)}")

    def _freeze(self, freeze_attn, freeze_ff, train_ln):
        """Freeze base weights. LayerNorm stays trainable if requested."""
        for name, param in self.model.named_parameters():
            nl = name.lower()
            # LayerNorm: trainable if requested
            if any(k in nl for k in LN_KEYS) and train_ln:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        out = self.model(inputs_embeds=x)
        return out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]

# ============================================================================
# ACTIVITY PROJECTION (Classification Head)
# ============================================================================

class ActivityProjection(nn.Module):
    """Classification head using last token representation."""
    def __init__(self, d_model=768, num_classes=4, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        return self.head(x[:, -1, :])  # Last token

# ============================================================================
# FULL LLM4HAR MODEL v2
# ============================================================================

class LLM4HAR(nn.Module):
    """LLM4HAR v2: Cross-domain HAR with optional reprogramming layer,
    two-stage alignment support, and reconstruction auxiliary loss.

    Pipeline:
      InstanceNorm -> SensorEmbedding(+time indices) -> PE
      -> [ReprogrammingLayer (gated)] -> LLM -> Projection
                                              -> [ReconstructionHead (aux)]

    Two-stage training:
      Stage 1: get_alignment_loss() for next-patch prediction with FFT loss
      Stage 2: forward() for classification; forward_with_recon() adds aux loss
    """
    def __init__(self, num_channels=6, window_samples=120, segment_count=8,
                 num_classes=4, llm_name='gpt2', llm_layers_to_keep=4,
                 freeze_attention=True, freeze_feedforward=True, train_layernorm=True,
                 dropout=0.1,
                 use_reprogramming=False, n_prototypes=64, n_heads=8,
                 use_recon_aux=False):
        super().__init__()

        # Auto-detect d_model from LLM's native hidden size
        config = AutoConfig.from_pretrained(llm_name, trust_remote_code=True)
        d_model = config.hidden_size
        self.d_model = d_model
        self.use_reprogramming = use_reprogramming

        patch_length = window_samples // segment_count  # 15
        num_tokens = num_channels * segment_count       # 48

        self.instance_norm = SensorInstanceNorm(num_channels)
        self.sensor_embedding = SensorEmbedding(num_channels, segment_count,
                                                 patch_length, d_model)
        self.positional_encoding = LearnablePositionalEncoding(num_tokens, d_model)
        self.llm_backbone = LLMBackbone(
            llm_name, llm_layers_to_keep,
            freeze_attention, freeze_feedforward, train_layernorm,
            config=config
        )
        self.activity_projection = ActivityProjection(d_model, num_classes, dropout)

        # Optional reprogramming layer (Level 2a, 3)
        self.reprogramming = None
        if use_reprogramming:
            self.reprogramming = ReprogrammingLayer(
                d_model, n_prototypes=n_prototypes,
                n_heads=n_heads, dropout=dropout
            )

        # Alignment head (Level 2b, 3) — created on-demand
        self._alignment_head = None

        # Reconstruction head (Level 2b, 3) — eager init for optimizer compatibility
        self._recon_head = ReconstructionHead(d_model) if use_recon_aux else None

    def _get_sensor_embeddings(self, x):
        """Run through normalization, embedding, PE, and optional reprogramming.
        Returns the embeddings that will be fed to the LLM backbone."""
        x = self.instance_norm(x)              # (N, 120, 6)
        x = self.sensor_embedding(x)           # (N, 48, d_model)
        x = self.positional_encoding(x)        # (N, 48, d_model)
        if self.reprogramming is not None:
            x = self.reprogramming(x)          # (N, 48, d_model)
        return x

    def forward(self, x):
        """x: (batch, 120, 6) -> logits: (batch, num_classes)"""
        x = self._get_sensor_embeddings(x)     # (N, 48, d_model)
        x = self.llm_backbone(x)               # (N, 48, d_model)
        return self.activity_projection(x)     # (N, num_classes)

    def forward_with_recon(self, x):
        """Single forward pass returning (logits, recon_loss).
        Used during classification training when use_recon_aux=True.
        Avoids double computation vs calling forward() + get_reconstruction_loss().
        Args:
            x: (batch, 120, 6) raw IMU window
        Returns:
            (logits, recon_loss) — logits: (batch, num_classes), recon_loss: scalar
        """
        sensor_emb = self._get_sensor_embeddings(x)   # (N, 48, d_model)
        llm_out = self.llm_backbone(sensor_emb)        # (N, 48, d_model)
        logits = self.activity_projection(llm_out)     # (N, num_classes)
        recon_loss = self._recon_head(llm_out, sensor_emb)
        return logits, recon_loss

    def get_reconstruction_loss(self, x):
        """Compute reconstruction loss for standalone use.
        Args:
            x: (batch, 120, 6) raw IMU window
        Returns:
            Reconstruction loss (scalar)
        """
        sensor_emb = self._get_sensor_embeddings(x)
        llm_out = self.llm_backbone(sensor_emb)
        return self._recon_head(llm_out, sensor_emb)

    def get_alignment_loss(self, x):
        """Compute next-patch prediction loss for Stage 1 alignment.
        Creates alignment head on first call (lazy init).
        Args:
            x: (batch, 120, 6) raw IMU window
        Returns:
            Combined MSE + FFT loss (scalar)
        """
        if self._alignment_head is None:
            self._alignment_head = AlignmentHead(self.d_model).to(x.device)
        sensor_emb = self._get_sensor_embeddings(x)    # (N, 48, d_model)
        llm_out = self.llm_backbone(sensor_emb)        # (N, 48, d_model)
        return self._alignment_head(llm_out, sensor_emb)

    def get_param_stats(self):
        """Return trainable vs frozen parameter counts per component."""
        stats = {}
        total_t, total_f = 0, 0
        components = [
            ('instance_norm', self.instance_norm),
            ('sensor_embedding', self.sensor_embedding),
            ('positional_encoding', self.positional_encoding),
            ('llm_backbone', self.llm_backbone),
            ('activity_projection', self.activity_projection),
        ]
        if self.reprogramming is not None:
            components.insert(3, ('reprogramming', self.reprogramming))
        if self._alignment_head is not None:
            components.append(('alignment_head', self._alignment_head))
        if self._recon_head is not None:
            components.append(('recon_head', self._recon_head))

        for name, mod in components:
            t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            f = sum(p.numel() for p in mod.parameters() if not p.requires_grad)
            stats[name] = {'trainable': t, 'frozen': f}
            total_t += t; total_f += f
        stats['total_trainable'] = total_t
        stats['total_frozen'] = total_f
        stats['total_params'] = total_t + total_f
        stats['d_model'] = self.d_model
        return stats