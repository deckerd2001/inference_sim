"""
Factory for creating performance models.
"""

from typing import Optional
from .base import BasePerformanceModel
from .roofline import RooflinePerformanceModel
from .vllm_roofline import VLLMRooflineModel

from ..config import ModelSpec, ParallelismSpec
from ..xpu_spec import xPUSpec
from ..communication import TPCommunicationStrategy


def create_performance_model(
    model_type: str,
    model_spec: ModelSpec,
    xpu_spec: xPUSpec,
    parallelism_spec: ParallelismSpec,
    calibration_data_path: Optional[str] = None,
    tp_comm_strategy: Optional[TPCommunicationStrategy] = None
) -> BasePerformanceModel:
    """
    Create a performance model instance.
    
    Args:
        model_type: Type of model ("roofline" or "vllm_roofline")
        model_spec: Model specification
        xpu_spec: xPU specification
        parallelism_spec: Parallelism specification
        calibration_data_path: Path to calibration data (required for vllm_roofline)
        tp_comm_strategy: Optional TP communication strategy
        
    Returns:
        BasePerformanceModel instance
        
    Raises:
        ValueError: If model_type is invalid or calibration_data_path is missing
    """
    if model_type == "roofline":
        return RooflinePerformanceModel(
            model_spec=model_spec,
            xpu_spec=xpu_spec,
            parallelism_spec=parallelism_spec,
            tp_comm_strategy=tp_comm_strategy
        )
    
    elif model_type == "vllm_roofline":
        if calibration_data_path is None:
            raise ValueError(
                "calibration_data_path is required for vllm_roofline model type"
            )
        
        # Load calibration data (RooflineParameters)
        roofline_params = _load_roofline_parameters(calibration_data_path)
        
        return VLLMRooflineModel(
            model_spec=model_spec,
            xpu_spec=xpu_spec,
            parallelism_spec=parallelism_spec,
            roofline_params=roofline_params,
            tp_comm_strategy=tp_comm_strategy
        )
    
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Must be 'roofline' or 'vllm_roofline'"
        )


def _load_roofline_parameters(file_path: str):
    """
    Load RooflineParameters from JSON file.
    
    Args:
        file_path: Path to JSON file containing roofline parameters
        
    Returns:
        RooflineParameters object
    """
    import json
    from vllm_benchmarks.roofline_estimator import RooflineParameters
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle both direct RooflineParameters format and nested format
    if "roofline_params" in data:
        params_dict = data["roofline_params"]
    else:
        params_dict = data
    
    return RooflineParameters(
        prefill_effective_tflops=params_dict["prefill_effective_tflops"],
        prefill_effective_bandwidth_gbs=params_dict["prefill_effective_bandwidth_gbs"],
        prefill_comm_overhead_s=params_dict.get("prefill_comm_overhead_s", 0.0),
        decode_effective_tflops=params_dict["decode_effective_tflops"],
        decode_effective_bandwidth_gbs=params_dict["decode_effective_bandwidth_gbs"],
        decode_comm_overhead_s=params_dict.get("decode_comm_overhead_s", 0.0),
        xpu_name=params_dict.get("xpu_name", ""),
        model_name=params_dict.get("model_name", ""),
    )
