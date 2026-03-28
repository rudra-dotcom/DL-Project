import argparse

import torch
from timm import create_model

import model


def build_model(model_name, num_classes):
    kwargs = {
        'num_classes': num_classes,
        'pretrained': False,
    }
    if model_name.startswith('repvit_'):
        kwargs['distillation'] = False
    return create_model(model_name, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='Print intermediate feature map sizes.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--input-size', type=int, default=32)
    parser.add_argument('--num-classes', type=int, default=100)
    args = parser.parse_args()

    net = build_model(args.model, args.num_classes)
    net.eval()

    if not hasattr(net, 'features'):
        raise RuntimeError(f'{args.model} does not expose a .features module list')

    hooks = []
    feature_shapes = []
    for idx, layer in enumerate(net.features):
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, layer_idx=idx: feature_shapes.append(
                    (f'features.{layer_idx}', tuple(output.shape))
                )
            )
        )

    with torch.inference_mode():
        net(torch.randn(1, 3, args.input_size, args.input_size))

    for hook in hooks:
        hook.remove()

    print(f'Feature map shapes for {args.model} @ {args.input_size}x{args.input_size}')
    for name, shape in feature_shapes:
        print(f'{name}: {shape}')


if __name__ == '__main__':
    main()
