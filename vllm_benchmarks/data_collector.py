"""
Data collection and storage for vLLM benchmarks.
"""

import json
from pathlib import Path
from typing import List, Optional
from .schema import BenchmarkPoint, BenchmarkData


class BenchmarkDataCollector:
    """Collects and stores benchmark data."""
    
    def __init__(self, output_dir: str = "benchmark_data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_benchmark_data(
        self,
        benchmark_data: BenchmarkData,
        output_path: Optional[str] = None
    ) -> str:
        """
        Save benchmark data to JSON file.
        
        Args:
            benchmark_data: BenchmarkData object
            output_path: Optional custom path. If None, auto-generate.
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            # Auto-generate path: benchmark_data/raw/{xpu}/{model}_{date}.json
            xpu_dir = self.output_dir / benchmark_data.xpu_name
            xpu_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = xpu_dir / f"{benchmark_data.model_name}_{benchmark_data.collection_date}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict for JSON serialization
        data_dict = {
            "metadata": {
                "xpu": benchmark_data.xpu_name,
                "model": benchmark_data.model_name,
                "vllm_version": benchmark_data.vllm_version,
                "collection_date": benchmark_data.collection_date,
                "collector": benchmark_data.collector,
            },
            "benchmarks": [
                {
                    "model_name": b.model_name,
                    "xpu_name": b.xpu_name,
                    "tp_size": b.tp_size,
                    "batch_size": b.batch_size,
                    "seq_length": b.seq_length,
                    "kv_cache_length": b.kv_cache_length,
                    "prefill_time_ms": b.prefill_time_ms,
                    "decode_time_ms": b.decode_time_ms,
                    "num_samples": b.num_samples,
                    "std_dev": b.std_dev,
                    "vllm_version": b.vllm_version,
                    "measurement_timestamp": b.measurement_timestamp,
                    "metadata": b.metadata,
                }
                for b in benchmark_data.benchmarks
            ],
            "additional_metadata": benchmark_data.metadata,
        }
        
        with open(output_path, 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        return str(output_path)
    
    def load_benchmark_data(self, file_path: str) -> BenchmarkData:
        """
        Load benchmark data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            BenchmarkData object
        """
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
        
        metadata = data_dict["metadata"]
        benchmarks = [
            BenchmarkPoint(
                model_name=b["model_name"],
                xpu_name=b["xpu_name"],
                tp_size=b.get("tp_size", 1),
                batch_size=b["batch_size"],
                seq_length=b.get("seq_length", 0),
                kv_cache_length=b.get("kv_cache_length", 0),
                prefill_time_ms=b.get("prefill_time_ms"),
                decode_time_ms=b.get("decode_time_ms"),
                num_samples=b.get("num_samples", 1),
                std_dev=b.get("std_dev"),
                vllm_version=b.get("vllm_version"),
                measurement_timestamp=b.get("measurement_timestamp"),
                metadata=b.get("metadata", {}),
            )
            for b in data_dict["benchmarks"]
        ]
        
        return BenchmarkData(
            xpu_name=metadata["xpu"],
            model_name=metadata["model"],
            vllm_version=metadata["vllm_version"],
            collection_date=metadata["collection_date"],
            collector=metadata.get("collector", "vllm_benchmarks"),
            benchmarks=benchmarks,
            metadata=data_dict.get("additional_metadata", {}),
        )
