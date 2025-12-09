# Scripts Directory

Helper scripts for benchmark analysis.

## analyze_benchmark_results.py

Comprehensive benchmark results analyzer.

**Usage:**
```bash
python3 scripts/analyze_benchmark_results.py <results_dir>
```

**Example:**
```bash
python3 scripts/analyze_benchmark_results.py results/cluster_benchmark_20251208_215426/
```

**Environment Variables:**
- `MODEL`: Model name
- `XPUS`: xPU types tested
- `WORKLOAD_ARRIVAL_RATE`: Arrival rate (req/s)
- `SIMULATION_DURATION`: Duration (seconds)

**Output:**
- Configuration summary
- Workload requirements
- Performance & cost analysis table
- Recommended configurations
- Failed tests summary
- TP scaling efficiency analysis
