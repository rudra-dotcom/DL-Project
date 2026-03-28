#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def patch_first_convolution(model_path: Path, output_path: Path) -> None:
    spec = ct.models.MLModel(str(model_path)).get_spec()

    if spec.WhichOneof("Type") != "neuralNetwork":
        raise ValueError(f"Expected a neuralNetwork model, found {spec.WhichOneof('Type')!r}")

    if not spec.neuralNetwork.layers:
        raise ValueError("Model has no layers to patch.")

    first_layer = spec.neuralNetwork.layers[0]
    if first_layer.WhichOneof("layer") != "convolution":
        raise ValueError(f"Expected first layer to be convolution, found {first_layer.WhichOneof('layer')!r}")

    conv = first_layer.convolution
    if conv.nGroups != 1 or conv.kernelChannels != 3:
        raise ValueError("Expected a standard RGB convolution as the first layer.")

    weights = np.array(conv.weights.floatValue, dtype=np.float32)
    weights = weights.reshape(
        conv.outputChannels,
        conv.kernelChannels,
        conv.kernelSize[0],
        conv.kernelSize[1],
    )

    if conv.hasBias:
        bias = np.array(conv.bias.floatValue, dtype=np.float32)
    else:
        bias = np.zeros(conv.outputChannels, dtype=np.float32)

    # Fold ImageNet normalization into the first convolution so the Core ML
    # model behaves like the original PyTorch eval pipeline:
    #   x' = (x / 255 - mean) / std
    scale = 1.0 / (255.0 * IMAGENET_STD)
    offset = -IMAGENET_MEAN / IMAGENET_STD

    original_weights = weights.copy()
    for channel in range(3):
        weights[:, channel, :, :] *= scale[channel]
        bias += original_weights[:, channel, :, :].sum(axis=(1, 2)) * offset[channel]

    del conv.weights.floatValue[:]
    conv.weights.floatValue.extend(weights.reshape(-1).tolist())

    if conv.hasBias:
        del conv.bias.floatValue[:]
        conv.bias.floatValue.extend(bias.tolist())
    else:
        conv.hasBias = True
        conv.bias.floatValue.extend(bias.tolist())

    spec.description.metadata.shortDescription = (
        "RepViT-M1.1 with ImageNet normalization folded into the first convolution."
    )
    spec.description.metadata.versionString = "normalized-first-conv-v1"

    ct.utils.save_spec(spec, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch the RepViT Core ML model so it matches ImageNet eval preprocessing."
    )
    parser.add_argument("model_path", type=Path, help="Path to the source .mlmodel file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .mlmodel path. Defaults to patching the source model in place.",
    )
    args = parser.parse_args()

    output_path = args.output or args.model_path
    patch_first_convolution(args.model_path, output_path)
    print(f"Patched model written to {output_path}")


if __name__ == "__main__":
    main()
