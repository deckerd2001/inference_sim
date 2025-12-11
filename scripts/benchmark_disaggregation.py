#!/usr/bin/env python3
"""
Benchmark script for comparing aggregated vs disaggregated configurations.

Runs multiple configurations and outputs comparative results.
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime
import argparse


# Benchmark configurations
CONFIGS = {
    "aggregated_mi300x": {
        "name": "Aggregated MI300X",
        "args": [
            "--model", "llama2-70b",
            "--xpu", "mi300x",
            "--n-xpus-per-node", "8",
            "--tp", "8",
        ]
    },
    "aggregated_a100": {
        "name": "Aggregated A100",
        "args": [
            "--model", "llama2-70b",
            "--xpu", "a100-80gb",
            "--n-xpus-per-node", "8",
            "--tp", "8",
        ]
    },
    "disagg_a100_mi300x_100gbps": {
        "name": "Disaggregated A100→MI300X (100GB/s)",
        "args": [
            "--model", "llama2-70b",
            "--disaggregated",
            "--prefill-xpu", "a100-80gb",
            "--prefill-n-xpus", "4",
            "--prefill-tp", "4",
            "--decode-xpu", "mi300x",
            "--decode-n-xpus", "8",
            "--decode-tp", "8",
            "--transfer-bandwidth", "100",
        ]
    },
    "disagg_a100_mi300x_400gbps": {
        "name": "Disaggregated A100→MI300X (400GB/s)",
        "args": [
            "--model", "llama2-70b",
            "--disaggregated",
            "--prefill-xpu", "a100-80gb",
            "--prefill-n-xpus", "4",
            "--prefill-tp", "4",
            "--decode-xpu", "mi300x",
            "--decode-n-xpus", "8",
            "--decode-tp", "8",
            "--transfer-bandwidth", "400",
        ]
    },
    "disagg_h100_mi300x_400gbps": {
        "name": "Disaggregated H100→MI300X (400GB/s)",
        "args": [
            "--model", "llama2-70b",
            "--disaggregated",
            "--prefill-xpu", "h100-80gb",
            "--prefill-n-xpus", "4",
            "--prefill-tp", "4",
            "--decode-xpu", "mi300x",
            "--decode-n-xpus", "8",
            "--decode-tp", "8",
            "--transfer-bandwidth", "400",
        ]
    },
}


def run_simulation(config_name: str, config: dict, arrival_rate: float,
                   duration: float, warm_up: float, output_dir: Path):
    """Run a single simulation configuration."""
    print(f"\n{'='*70}")
    print(f"Running: {config['name']}")
    print(f"Arrival Rate: {arrival_rate} req/s")
    print(f"{'='*70}\n")

    # Build command
    cmd = [
        sys.executable, "-m", "llm_inference_simulator",
        *config["args"],
        "--arrival-rate", str(arrival_rate),
        "--duration", str(duration),
        "--warm-up", str(warm_up),
    ]

    # Output file
    output_file = output_dir / f"{config_name}_rate{arrival_rate}.json"
    cmd.extend(["--output", str(output_file)])

    try:
        # Run simulation
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)

        # Load and return results
        with open(output_file, 'r') as f:
            return json.load(f)

    except subprocess.CalledProcessError as e:
        print(f"❌ Simulation failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None


def format_results_table(results: dict):
    """Format results as a comparison table."""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*100)

    # Header
    print(f"\n{'Configuration':<40} {'Rate':<8} {'Throughput':<12} {'TTFT':<10} {'E2E':<10} {'Cost/Hour':<12} {'Tok/$':<12}")
    print("-"*100)

    # Sort by arrival rate then config name
    sorted_results = sorted(results.items(),
                          key=lambda x: (x[1]['arrival_rate'], x[0]))

    for result_key, data in sorted_results:
        # Get actual config name from data (not the result key which includes rate)
        config_name = data.get('config_name', result_key.rsplit('_rate', 1)[0])
        name = CONFIGS[config_name]['name'][:38]
        rate = data['arrival_rate']
        throughput = data.get('throughput', {}).get('tokens_per_sec', 0)
        ttft = data.get('first_token_latency', {}).get('mean', 0)
        e2e = data.get('end_to_end_latency', {}).get('mean', 0)

        # Calculate cost (simplified - would need to parse from output)
        cost = data.get('cost_per_hour', 0)
        tok_per_dollar = (throughput * 3600 / cost) if cost > 0 else 0

        print(f"{name:<40} {rate:<8.1f} {throughput:<12.1f} {ttft:<10.3f} {e2e:<10.3f} ${cost:<11.2f} {tok_per_dollar:<12,.0f}")

    print("="*100)


def generate_csv_report(results: dict, output_file: Path):
    """Generate CSV report of results."""
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Config', 'Arrival_Rate', 'Throughput_TokS', 'TTFT_s', 'E2E_Latency_s',
            'Total_Requests', 'Completed_Requests', 'xPU_Util',
            'Peak_Memory_GB', 'Cost_Hour', 'Tokens_Dollar'
        ])

        # Data rows
        for result_key, data in sorted(results.items()):
            # Get actual config name from data (not the result key which includes rate)
            config_name = data.get('config_name', result_key.rsplit('_rate', 1)[0])
            name = CONFIGS[config_name]['name']
            rate = data['arrival_rate']
            throughput = data.get('throughput', {}).get('tokens_per_sec', 0)
            ttft = data.get('first_token_latency', {}).get('mean', 0)
            e2e = data.get('end_to_end_latency', {}).get('mean', 0)
            total_req = data.get('total_requests', 0)
            completed = data.get('completed_requests', 0)
            xpu_util = data.get('xpu_utilization', 0)
            peak_mem = data.get('memory', {}).get('peak_memory_gb', 0)
            cost = data.get('cost_per_hour', 0)
            tok_per_dollar = (throughput * 3600 / cost) if cost > 0 else 0

            writer.writerow([
                name, rate, throughput, ttft, e2e,
                total_req, completed, xpu_util,
                peak_mem, cost, tok_per_dollar
            ])

    print(f"\n✅ CSV report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark disaggregation configurations')
    parser.add_argument('--arrival-rates', nargs='+', type=float, default=[0.5, 1.0, 1.5, 2.0],
                       help='Arrival rates to test (default: 0.5 1.0 1.5 2.0)')
    parser.add_argument('--duration', type=float, default=20.0,
                       help='Simulation duration in seconds (default: 20)')
    parser.add_argument('--warm-up', type=float, default=5.0,
                       help='Warm-up duration in seconds (default: 5)')
    parser.add_argument('--configs', nargs='+', choices=list(CONFIGS.keys()),
                       default=list(CONFIGS.keys()),
                       help='Configurations to test (default: all)')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark_results'),
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Starting benchmark suite")
    print(f"   Configurations: {len(args.configs)}")
    print(f"   Arrival rates: {args.arrival_rates}")
    print(f"   Output: {output_dir}")

    # Run all configurations
    all_results = {}

    for config_name in args.configs:
        if config_name not in CONFIGS:
            print(f"⚠️  Unknown config: {config_name}")
            continue

        config = CONFIGS[config_name]

        for arrival_rate in args.arrival_rates:
            result_key = f"{config_name}_rate{arrival_rate}"

            result = run_simulation(
                config_name, config, arrival_rate,
                args.duration, args.warm_up, output_dir
            )

            if result:
                result['config_name'] = config_name
                result['arrival_rate'] = arrival_rate
                all_results[result_key] = result

    # Generate reports
    if all_results:
        format_results_table(all_results)

        csv_file = output_dir / 'benchmark_summary.csv'
        generate_csv_report(all_results, csv_file)

        # Save full results as JSON
        json_file = output_dir / 'benchmark_full_results.json'
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✅ Full results saved to: {json_file}")

    print(f"\n✨ Benchmark complete!")
    print(f"   Results directory: {output_dir}")


if __name__ == '__main__':
    main()