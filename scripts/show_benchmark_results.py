#!/usr/bin/env python3
"""
Quick script to display comparison table from existing benchmark results.
Usage: python3 show_benchmark_results.py benchmark_results/20251209_172146
"""

import json
import sys
from pathlib import Path
import re


def parse_results(results_dir: Path):
    """Parse all JSON files in the results directory."""
    results = {}

    for json_file in results_dir.glob('*.json'):
        if json_file.name.startswith('benchmark_'):
            continue  # Skip summary files

        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract config name and rate from filename
        # Format: configname_rateX.X.json
        match = re.match(r'(.+)_rate([\d.]+)\.json', json_file.name)
        if match:
            config_name = match.group(1)
            rate = float(match.group(2))

            results[json_file.stem] = {
                'config_name': config_name,
                'arrival_rate': rate,
                'data': data
            }

    return results


def format_table(results: dict):
    """Format results as a comparison table."""

    config_names = {
        'aggregated_mi300x': 'Aggregated MI300X',
        'aggregated_a100': 'Aggregated A100',
        'disagg_a100_mi300x_100gbps': 'Disagg A100→MI300X (100GB/s)',
        'disagg_a100_mi300x_400gbps': 'Disagg A100→MI300X (400GB/s)',
        'disagg_h100_mi300x_400gbps': 'Disagg H100→MI300X (400GB/s)',
    }

    print("\n" + "="*110)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*110)

    # Header
    print(f"\n{'Configuration':<40} {'Rate':<8} {'Throughput':<12} {'TTFT':<10} {'E2E':<10} {'Util':<8} {'Completed':<10}")
    print("-"*110)

    # Sort by config name then rate
    sorted_results = sorted(results.items(),
                          key=lambda x: (x[1]['config_name'], x[1]['arrival_rate']))

    for key, item in sorted_results:
        config_name = item['config_name']
        name = config_names.get(config_name, config_name)[:38]
        rate = item['arrival_rate']
        data = item['data']

        throughput = data.get('throughput', {}).get('tokens_per_sec', 0)
        ttft = data.get('first_token_latency', {}).get('mean', 0)
        e2e = data.get('end_to_end_latency', {}).get('mean', 0)
        util = data.get('xpu_utilization', 0)
        completed = data.get('completed_requests', 0)

        print(f"{name:<40} {rate:<8.1f} {throughput:<12.1f} {ttft:<10.4f} {e2e:<10.4f} {util:<8.1%} {completed:<10}")

    print("="*110)

    # Summary
    print("\n📊 Key Insights:")

    # Group by config
    by_config = {}
    for key, item in results.items():
        config = item['config_name']
        if config not in by_config:
            by_config[config] = []
        by_config[config].append(item)

    # Compare throughput at same load
    print("\n🔄 Throughput at rate=1.5:")
    for config, items in sorted(by_config.items()):
        name = config_names.get(config, config)
        for item in items:
            if abs(item['arrival_rate'] - 1.5) < 0.01:
                throughput = item['data'].get('throughput', {}).get('tokens_per_sec', 0)
                print(f"  {name:<45} {throughput:>8.1f} tok/s")

    # Compare latency
    print("\n⏱️  TTFT at rate=1.0:")
    for config, items in sorted(by_config.items()):
        name = config_names.get(config, config)
        for item in items:
            if abs(item['arrival_rate'] - 1.0) < 0.01:
                ttft = item['data'].get('first_token_latency', {}).get('mean', 0)
                print(f"  {name:<45} {ttft:>8.4f}s")

    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 show_benchmark_results.py <results_directory>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"❌ Directory not found: {results_dir}")
        sys.exit(1)

    results = parse_results(results_dir)

    if not results:
        print(f"❌ No results found in {results_dir}")
        sys.exit(1)

    format_table(results)


if __name__ == '__main__':
    main()