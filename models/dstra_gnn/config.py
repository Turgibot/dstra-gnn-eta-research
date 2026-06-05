"""DSTRA-GNN variant configurations (A1–A6)."""
from dataclasses import dataclass, field


@dataclass
class DSTRAConfig:
    # Variant identity
    variant: str = "temporal_route_aware"   # A6 full model

    # Architecture toggles
    use_dynamic_edges: bool = True   # traversal + interaction edges
    use_temporal:      bool = True   # Transformer temporal aggregator
    use_route:         bool = True   # route-left encoder

    # Graph encoder (GATv2)
    hidden_dim:   int = 128
    gat_heads:    int = 4
    gat_layers:   int = 2
    dropout:      float = 0.1

    # Temporal aggregator
    temporal_heads:  int = 4
    temporal_layers: int = 2
    window_size:     int = 30   # H snapshots per window

    # Route encoder
    route_embed_dim: int = 64

    # MoE head
    moe_experts:     int = 6
    moe_top_k:       int = 2

    # Training
    lr:           float = 1e-3
    weight_decay: float = 1e-2
    epochs:       int   = 200
    patience:     int   = 20
    batch_size:   int   = 1    # windows per gradient step
    warmup_steps: int   = 1000


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
}
