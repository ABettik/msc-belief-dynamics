from typing import List, Optional, Literal, Dict, Sequence
from dataclasses import dataclass

@dataclass
class OptimizerArgs:
    """
    Hyperparameters for the optimizer and scheduler.
    """
    lr: float                # Learning rate for AdamW
    weight_decay: float                # Weight decay for AdamW
    eps: float               # Epsilon for AdamW

@dataclass
class LrSchedulerArgs:
    """
    Hyperparameters specifically for the learning-rate scheduler.
    """
    max_lr: float            # Maximum learning rate for OneCycleLR
    pct_start: float         # Fraction of cycle for warmup
    div_factor: float        # Divider factor matching optimizer.div_factor

@dataclass
class MaskerConfig:
    force_active: bool
    mode: Literal['temporal', 'neuron', 'all', 'timestep_only']
    ratio: float
    zero_ratio: float
    random_ratio: float
    expand_prob: float
    max_timespan: int
    channels: Optional[List[int]]
    timesteps: Optional[List[int]]
    mask_regions: List[str]
    target_regions: List[str]
    n_mask_regions: int

@dataclass
class ContextConfig:
    forward: int  # Max future bins (-1 = unlimited)
    backward: int # Max past bins (-1 = unlimited)

@dataclass
class NormAndNoiseConfig:
    active: bool
    smooth_sd: float
    norm: Literal['zscore']
    eps: float
    white_noise_sd: float
    constant_offset_sd: float

@dataclass
class EmbedderStackConfig:
    active: bool
    size: int
    stride: int

@dataclass
class EmbedderConfig:
    n_channels: int
    n_blocks: int
    n_dates: int
    use_prompt: bool
    use_session: bool
    max_F: int
    mode: Literal['linear']
    mult: int
    adapt: bool
    pos: bool
    act: Literal['softsign']
    scale: float
    bias: bool
    dropout: float
    fixup_init: bool
    init_range: float
    spike_log_init: bool
    max_spikes: int
    tokenize_binary_mask: bool
    stack: EmbedderStackConfig

@dataclass
class TransformerConfig:
    n_layers: int
    hidden_size: int
    use_scalenorm: bool
    use_rope: bool
    rope_theta: float
    n_heads: int
    attention_bias: bool
    act: Literal['gelu']
    inter_size: int
    mlp_bias: bool
    dropout: float
    fixup_init: bool

@dataclass
class FactorsConfig:
    active: bool
    size: int
    act: Literal['relu']
    bias: bool
    dropout: float
    fixup_init: bool
    init_range: float

@dataclass
class EncoderConfig:
    from_pt: Optional[str]
    stitching: bool
    masker: MaskerConfig
    context: ContextConfig
    norm_and_noise: NormAndNoiseConfig
    embedder: EmbedderConfig
    transformer: TransformerConfig
    factors: FactorsConfig

@dataclass
class DecoderConfig:
    from_pt: Optional[str]

@dataclass
class NdtConfig:
    encoder: EncoderConfig
    decoder: DecoderConfig

@dataclass
class NDT1SSLConfig:
    ndt_cfg: NdtConfig
    optimizer_args: OptimizerArgs
    lr_scheduler_args: LrSchedulerArgs
    method_name: str = "ssl"
    num_neurons: int = None
    tc_weight: float = 1.0
    prpd_weight: float = 1.0
    prpd_head_hidden_dims: Sequence[int] = None
    latent_dim: int = 16
