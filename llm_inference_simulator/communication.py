"""
Communication patterns and collective operations for distributed inference.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class CollectiveOp(Enum):
    """Types of collective communication operations."""
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"
    REDUCE = "reduce"
    NONE = "none"


class CommunicationAlgorithm(Enum):
    """Algorithms for collective operations."""
    RING = "ring"
    TREE = "tree"
    HIERARCHICAL = "hierarchical"
    NCCL_AUTO = "nccl_auto"


@dataclass
class CommunicationPattern:
    """
    Defines communication pattern for a specific operation.

    In Tensor Parallelism:
    - Attention output: typically ALL_REDUCE
    - MLP output: typically ALL_REDUCE
    - Forward: ALL_GATHER weights (disaggregated) or ALL_REDUCE results (aggregated)
    """

    # Operation type
    collective_op: CollectiveOp = CollectiveOp.ALL_REDUCE

    # Algorithm to use
    algorithm: CommunicationAlgorithm = CommunicationAlgorithm.RING

    # Whether to overlap communication with computation
    overlap_compute: bool = False

    # Custom bandwidth multiplier (for tuning)
    bandwidth_multiplier: float = 1.0


def _default_comm_pattern() -> CommunicationPattern:
    """Default communication pattern factory."""
    return CommunicationPattern(
        collective_op=CollectiveOp.ALL_REDUCE,
        algorithm=CommunicationAlgorithm.RING,
    )


@dataclass
class TPCommunicationStrategy:
    """
    Communication strategy for Tensor Parallelism.

    Disaggregation: Each GPU holds a shard of weights, compute independently
    Aggregation: Combine results from all GPUs
    """

    # QKV projection: typically column-parallel (all-gather) or row-parallel (reduce-scatter)
    qkv_projection: CommunicationPattern = field(default_factory=_default_comm_pattern)

    # Attention output: typically all-reduce
    attention_output: CommunicationPattern = field(default_factory=_default_comm_pattern)

    # MLP up projection
    mlp_up_projection: CommunicationPattern = field(default_factory=_default_comm_pattern)

    # MLP down projection
    mlp_down_projection: CommunicationPattern = field(default_factory=_default_comm_pattern)

    # Use disaggregated forward pass (weights sharded across GPUs)
    use_disaggregated_forward: bool = True

    # Use aggregated backward pass (if training, not relevant for inference)
    use_aggregated_backward: bool = False


def estimate_collective_time(
    collective_op: CollectiveOp,
    data_size_bytes: float,
    num_devices: int,
    bandwidth_gbs: float,
    latency_us: float,
    algorithm: CommunicationAlgorithm = CommunicationAlgorithm.RING,
) -> float:
    """
    Estimate time for a collective operation.

    Args:
        collective_op: Type of collective operation
        data_size_bytes: Size of data to communicate
        num_devices: Number of participating devices
        bandwidth_gbs: Bandwidth in GB/s
        latency_us: Latency in microseconds
        algorithm: Communication algorithm

    Returns:
        Estimated time in seconds
    """
    bandwidth_bs = bandwidth_gbs * 1e9  # Convert to bytes/sec
    latency_s = latency_us * 1e-6  # Convert to seconds

    if num_devices <= 1:
        return 0.0

    if collective_op == CollectiveOp.NONE or data_size_bytes <= 0:
        return 0.0

    if algorithm == CommunicationAlgorithm.RING:
        if collective_op == CollectiveOp.ALL_REDUCE:
            # Ring all-reduce: 2 * (N-1) / N * data_size / bandwidth
            transfer_time = 2 * (num_devices - 1) / num_devices * data_size_bytes / bandwidth_bs
            return transfer_time + latency_s

        elif collective_op == CollectiveOp.ALL_GATHER:
            # All-gather: (N-1) / N * data_size / bandwidth
            transfer_time = (num_devices - 1) / num_devices * data_size_bytes / bandwidth_bs
            return transfer_time + latency_s

        elif collective_op == CollectiveOp.REDUCE_SCATTER:
            # Reduce-scatter: (N-1) / N * data_size / bandwidth
            transfer_time = (num_devices - 1) / num_devices * data_size_bytes / bandwidth_bs
            return transfer_time + latency_s

        elif collective_op == CollectiveOp.ALL_TO_ALL:
            # All-to-all: (N-1) / N * data_size / bandwidth
            transfer_time = (num_devices - 1) / num_devices * data_size_bytes / bandwidth_bs
            return transfer_time + latency_s

        elif collective_op == CollectiveOp.BROADCAST:
            # Broadcast: data_size / bandwidth (approximately)
            transfer_time = data_size_bytes / bandwidth_bs
            return transfer_time + latency_s

        elif collective_op == CollectiveOp.REDUCE:
            # Reduce: similar to reduce-scatter
            transfer_time = (num_devices - 1) / num_devices * data_size_bytes / bandwidth_bs
            return transfer_time + latency_s

    elif algorithm == CommunicationAlgorithm.TREE:
        # Tree-based algorithms: log(N) steps
        import math
        steps = math.ceil(math.log2(num_devices))
        transfer_time = steps * data_size_bytes / bandwidth_bs
        return transfer_time + latency_s * steps

    elif algorithm == CommunicationAlgorithm.HIERARCHICAL:
        # Simplified hierarchical: similar to ring but with some optimization
        transfer_time = 1.5 * (num_devices - 1) / num_devices * data_size_bytes / bandwidth_bs
        return transfer_time + latency_s

    # Default fallback (NCCL_AUTO or unknown)
    transfer_time = 2 * (num_devices - 1) / num_devices * data_size_bytes / bandwidth_bs
    return transfer_time + latency_s


def create_megatron_tp_strategy() -> TPCommunicationStrategy:
    """
    Create Megatron-LM style TP communication strategy.

    Column-parallel for first linear, row-parallel for second linear.
    """
    return TPCommunicationStrategy(
        qkv_projection=CommunicationPattern(
            collective_op=CollectiveOp.NONE,
            algorithm=CommunicationAlgorithm.RING,
        ),

        attention_output=CommunicationPattern(
            collective_op=CollectiveOp.ALL_REDUCE,
            algorithm=CommunicationAlgorithm.RING,
        ),
        mlp_up_projection=CommunicationPattern(
            collective_op=CollectiveOp.NONE,
            algorithm=CommunicationAlgorithm.RING,
        ),
        mlp_down_projection=CommunicationPattern(
            collective_op=CollectiveOp.ALL_REDUCE,
            algorithm=CommunicationAlgorithm.RING,
        ),
        use_disaggregated_forward=True,
    )


def create_sequence_parallel_strategy() -> TPCommunicationStrategy:
    """
    Create sequence parallelism communication strategy.

    Shard along sequence dimension in addition to tensor dimension.
    """
    return TPCommunicationStrategy(
        qkv_projection=CommunicationPattern(
            collective_op=CollectiveOp.ALL_GATHER,
            algorithm=CommunicationAlgorithm.RING,
        ),
        attention_output=CommunicationPattern(
            collective_op=CollectiveOp.REDUCE_SCATTER,
            algorithm=CommunicationAlgorithm.RING,
        ),
        mlp_up_projection=CommunicationPattern(
            collective_op=CollectiveOp.ALL_GATHER,
            algorithm=CommunicationAlgorithm.RING,
        ),
        mlp_down_projection=CommunicationPattern(
            collective_op=CollectiveOp.REDUCE_SCATTER,
            algorithm=CommunicationAlgorithm.RING,
        ),
        use_disaggregated_forward=True,
    )
