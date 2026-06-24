"""DSTRA-GNN variant configurations (A1–A6)."""
from dataclasses import dataclass


@dataclass
class DSTRAConfig:
    # Variant identity
    variant: str = "temporal_route_aware"

    # Architecture toggles
    use_dynamic_edges: bool = True
    use_temporal:      bool = True
    use_route:         bool = True
    temporal_type:     str  = "transformer"   # "transformer" | "gru"

    # Graph encoder (GATv2)
    hidden_dim:   int   = 128
    gat_heads:    int   = 4
    gat_layers:   int   = 2
    dropout:      float = 0.1

    # Temporal aggregator
    temporal_heads:  int = 4
    temporal_layers: int = 2
    window_size:     int = 30

    # Route encoder
    route_embed_dim: int = 64

    # Fusion MLP
    fusion_out: int = 192   # input dim to MoE head

    # MoE head
    moe_experts:  int   = 6
    moe_top_k:    int   = 2
    moe_noise:    float = 0.15   # Gaussian noise std on router logits during training
    moe_temp:     float = 1.2    # Temperature divisor on router logits

    # Training
    lr:             float = 1e-3
    weight_decay:   float = 1e-2
    epochs:         int   = 500
    patience:       int   = 50
    batch_size:     int   = 6    # windows per gradient step
    warmup_epochs:  int   = 2    # linear LR warmup in epochs

    # Loss weights
    lb_weight:      float = 0.05    # MoE load-balancing loss coefficient
    entropy_weight: float = 0.002   # Router KL-to-uniform entropy coefficient


VARIANTS = {
    "A1": DSTRAConfig(variant="base_graph",
                      use_dynamic_edges=False, use_temporal=False, use_route=False),
    "A2": DSTRAConfig(variant="dynamic_graph",
                      use_dynamic_edges=True,  use_temporal=False, use_route=False),
    "A3": DSTRAConfig(variant="route_aware_graph",
                      use_dynamic_edges=True,  use_temporal=False, use_route=True),
    "A4": DSTRAConfig(variant="temporal_base",
                      use_dynamic_edges=False, use_temporal=True,  use_route=False),
    "A5": DSTRAConfig(variant="temporal_dynamic",
                      use_dynamic_edges=True,  use_temporal=True,  use_route=False),
    "A6": DSTRAConfig(variant="temporal_route_aware",
                      use_dynamic_edges=True,  use_temporal=True,  use_route=True),
    "A4_GRU": DSTRAConfig(variant="temporal_base_gru",
                           use_dynamic_edges=False, use_temporal=True, use_route=False,
                           temporal_type="gru"),
    "A5_GRU": DSTRAConfig(variant="temporal_dynamic_gru",
                           use_dynamic_edges=True, use_temporal=True, use_route=False,
                           temporal_type="gru"),
    "A6_GRU": DSTRAConfig(variant="temporal_route_aware_gru",
                           use_dynamic_edges=True, use_temporal=True, use_route=True,
                           temporal_type="gru"),
}
