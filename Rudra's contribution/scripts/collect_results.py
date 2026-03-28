import argparse
import json
from pathlib import Path

import pandas as pd


def load_benchmarks(benchmark_dir):
    if not benchmark_dir:
        return {}
    benchmarks = {}
    for bench_path in Path(benchmark_dir).glob('*.json'):
        benchmarks[bench_path.stem] = json.loads(bench_path.read_text())
    return benchmarks


def collect_runs(run_root, benchmarks):
    summary_rows = []
    curve_rows = []

    for log_path in Path(run_root).rglob('log.txt'):
        run_dir = log_path.parent
        args_path = run_dir / 'args.txt'
        if not args_path.exists():
            continue

        args_data = json.loads(args_path.read_text())
        model_name = args_data.get('model', run_dir.parent.name)
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        if not records:
            continue

        for record in records:
            curve_rows.append({
                'model': model_name,
                'run_dir': str(run_dir),
                **record,
            })

        best_record = max(records, key=lambda item: item.get('test_acc1', float('-inf')))
        final_record = records[-1]
        bench_data = benchmarks.get(model_name, {})
        params_m = bench_data.get('params_m')
        if params_m is None and final_record.get('n_parameters') is not None:
            params_m = final_record.get('n_parameters') / 1e6
        summary_rows.append({
            'model': model_name,
            'run_dir': str(run_dir),
            'epochs': args_data.get('epochs'),
            'batch_size': args_data.get('batch_size'),
            'lr': args_data.get('lr'),
            'best_acc1': best_record.get('test_acc1'),
            'best_epoch': best_record.get('epoch'),
            'final_acc1': final_record.get('test_acc1'),
            'final_train_loss': final_record.get('train_loss'),
            'n_parameters': final_record.get('n_parameters'),
            'params_m': params_m,
            'flops_g': bench_data.get('flops_g'),
            'latency_mean_ms': bench_data.get('latency_mean_ms'),
            'latency_median_ms': bench_data.get('latency_median_ms'),
            'throughput_images_per_s': bench_data.get('throughput_images_per_s'),
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(curve_rows)


def main():
    parser = argparse.ArgumentParser(description='Collect training logs into CSV summaries.')
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--benchmark-dir', default='')
    parser.add_argument('--summary-output', default='results/generated/cifar100_summary.csv')
    parser.add_argument('--curve-output', default='results/generated/cifar100_curves.csv')
    args = parser.parse_args()

    benchmarks = load_benchmarks(args.benchmark_dir)
    summary_df, curve_df = collect_runs(args.run_root, benchmarks)

    summary_path = Path(args.summary_output)
    curve_path = Path(args.curve_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_df.empty:
        summary_df.to_csv(summary_path, index=False)
        curve_df.to_csv(curve_path, index=False)
        print(f'No completed runs found under {args.run_root}')
        return

    summary_df.sort_values(by='best_acc1', ascending=False).to_csv(summary_path, index=False)
    curve_df.to_csv(curve_path, index=False)

    print(f'Wrote summary CSV to {summary_path}')
    print(f'Wrote curve CSV to {curve_path}')


if __name__ == '__main__':
    main()
