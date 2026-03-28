import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from fvcore.nn import FlopCountAnalysis
from timm import create_model

import model
import utils


def build_model(model_name, num_classes):
    kwargs = {
        'num_classes': num_classes,
        'pretrained': False,
    }
    if model_name.startswith('repvit_'):
        kwargs['distillation'] = False
    return create_model(model_name, **kwargs)


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def measure_latency(model, device, input_size, batch_size, warmup_steps, steps):
    inputs = torch.randn(batch_size, 3, input_size, input_size, device=device)
    with torch.inference_mode():
        for _ in range(warmup_steps):
            model(inputs)
        synchronize(device)

        timings = []
        for _ in range(steps):
            start = time.perf_counter()
            model(inputs)
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    return {
        'latency_batch_size': batch_size,
        'latency_mean_ms': sum(timings) / len(timings),
        'latency_median_ms': timings[len(timings) // 2],
        'latency_p90_ms': timings[int(len(timings) * 0.9)],
    }


def measure_throughput(model, device, input_size, batch_size, warmup_steps, steps):
    inputs = torch.randn(batch_size, 3, input_size, input_size, device=device)
    with torch.inference_mode():
        for _ in range(warmup_steps):
            model(inputs)
        synchronize(device)

        total_images = 0
        start = time.perf_counter()
        for _ in range(steps):
            model(inputs)
            total_images += batch_size
        synchronize(device)
        elapsed = time.perf_counter() - start
    return {
        'throughput_batch_size': batch_size,
        'throughput_images_per_s': total_images / elapsed,
    }


def compute_model_stats(model, input_size):
    dummy_input = torch.randn(1, 3, input_size, input_size)
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    try:
        flops_g = FlopCountAnalysis(model.cpu(), dummy_input).total() / 1e9
    except Exception:
        flops_g = None
    return params_m, flops_g


def main():
    parser = argparse.ArgumentParser(description='Benchmark latency and throughput.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--input-size', type=int, default=32)
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--latency-batch-size', type=int, default=1)
    parser.add_argument('--throughput-batch-size', type=int, default=256)
    parser.add_argument('--warmup-steps', type=int, default=50)
    parser.add_argument('--latency-steps', type=int, default=200)
    parser.add_argument('--throughput-steps', type=int, default=100)
    parser.add_argument('--fuse-bn', action='store_true')
    parser.add_argument('--output', default='')
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    bench_model = build_model(args.model, args.num_classes)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        bench_model.load_state_dict(checkpoint_state_dict(checkpoint), strict=False)
    if args.fuse_bn:
        utils.replace_batchnorm(bench_model)
    bench_model.eval()

    params_m, flops_g = compute_model_stats(build_model(args.model, args.num_classes), args.input_size)
    bench_model.to(device)

    latency = measure_latency(
        bench_model, device, args.input_size, args.latency_batch_size, args.warmup_steps, args.latency_steps
    )
    throughput = measure_throughput(
        bench_model,
        device,
        args.input_size,
        args.throughput_batch_size,
        max(10, args.warmup_steps // 2),
        args.throughput_steps,
    )

    payload = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'device': args.device,
        'input_size': args.input_size,
        'num_classes': args.num_classes,
        'params_m': params_m,
        'flops_g': flops_g,
        **latency,
        **throughput,
    }
    print(json.dumps(payload, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
