"""
Cluster abstraction for LLM inference scheduling.

This module provides different cluster implementations:
- AggregatedCluster: Single GPU pool (prefill and decode share resources)
- DisaggregatedCluster: Separate prefill and decode clusters
- ContinuousBatchingCluster: vLLM-style continuous batching (future)
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .request import Request
from .events import Event


@dataclass
class ScheduleResult:
    """Result from scheduling attempt."""
    busy_time: float  # How long GPU will be busy
    event: Event      # Event to schedule when done


class BaseCluster(ABC):
    """
    Abstract base class for cluster implementations.

    Different cluster types implement different scheduling policies:
    - Aggregated: Decode priority, mutually exclusive prefill/decode
    - Disaggregated: Independent prefill and decode clusters
    - Continuous: vLLM-style iteration-level scheduling
    """

    @abstractmethod
    def try_schedule(self, current_time: float) -> List[ScheduleResult]:
        """
        Try to schedule work.

        Returns:
            List of ScheduleResult for all scheduled work.
            Empty list if nothing can be scheduled.
        """
        pass

    @abstractmethod
    def is_busy(self) -> bool:
        """Check if cluster has any busy resources."""
        pass

    @abstractmethod
    def add_request(self, request: Request):
        """Add a new request to the cluster."""
        pass

    @abstractmethod
    def handle_prefill_finished(self, batch_id: int, current_time: float):
        """Handle prefill completion."""
        pass

    @abstractmethod
    def handle_decode_step_finished(self, batch_id: int, current_time: float) -> Optional[ScheduleResult]:
        """
        Handle decode step completion.

        Returns:
            ScheduleResult if batch continues, None if batch finished
        """
        pass

    @abstractmethod
    def get_memory_usage(self):
        """Get current memory usage statistics."""
        pass


class ClusterFactory:
    """Factory for creating clusters."""

    @staticmethod
    def create(config) -> BaseCluster:
        """
        Create appropriate cluster based on config.

        Args:
            config: SimulatorConfig

        Returns:
            Cluster instance
        """
        # Import here to avoid circular dependency
        from .aggregated_cluster import AggregatedCluster
        from .disaggregated_cluster import DisaggregatedCluster

        if config.is_disaggregated:
            # Disaggregated mode
            disagg_spec = config.disaggregation_spec

            return DisaggregatedCluster(
                model_spec=config.model_spec,
                prefill_xpu_spec=disagg_spec.prefill_cluster.xpu_spec,
                prefill_n_xpus=disagg_spec.prefill_cluster.total_xpus,
                prefill_parallelism=disagg_spec.prefill_parallelism,
                decode_xpu_spec=disagg_spec.decode_cluster.xpu_spec,
                decode_n_xpus=disagg_spec.decode_cluster.total_xpus,
                decode_parallelism=disagg_spec.decode_parallelism,
                transfer_bandwidth_gbs=disagg_spec.transfer_bandwidth_gbs,
                scheduler_spec=config.scheduler_spec
            )
        else:
            # Aggregated mode - use cluster_spec, not hardware_spec!
            return AggregatedCluster(
                model_spec=config.model_spec,
                xpu_spec=config.cluster_spec.xpu_spec,
                n_xpus=config.cluster_spec.total_xpus,
                parallelism_spec=config.parallelism_spec,
                scheduler_spec=config.scheduler_spec
            )