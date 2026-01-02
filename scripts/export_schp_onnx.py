#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DatasetSettings:
    name: str
    num_classes: int
    input_size_hw: tuple[int, int]


DATASETS: dict[str, DatasetSettings] = {
    "lip": DatasetSettings(name="lip", num_classes=20, input_size_hw=(473, 473)),
    "atr": DatasetSettings(name="atr", num_classes=18, input_size_hw=(512, 512)),
    "pascal": DatasetSettings(name="pascal", num_classes=7, input_size_hw=(512, 512)),
}


# === Minimal SCHP model definition (inference/export only) ===
# The original SCHP repo uses InPlaceABNSync which compiles a C++/CUDA extension.
# For ONNX export on CPU-only machines, this script uses a pure-PyTorch ABN equivalent
# with the same parameter/buffer names so official checkpoints load cleanly.


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        return torch, nn, F
    except Exception as exc:
        raise RuntimeError("torch is required to export SCHP to ONNX") from exc


ACT_RELU = "relu"
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"


def build_schp_resnet101(num_classes: int):
    torch, nn, F = _import_torch()
    import functools

    class ABN(nn.Module):
        def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            activation: str = ACT_LEAKY_RELU,
            slope: float = 0.01,
        ) -> None:
            super().__init__()
            self.num_features = int(num_features)
            self.affine = bool(affine)
            self.eps = float(eps)
            self.momentum = float(momentum)
            self.activation = str(activation)
            self.slope = float(slope)
            if self.affine:
                self.weight = nn.Parameter(torch.ones(self.num_features))
                self.bias = nn.Parameter(torch.zeros(self.num_features))
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)
            self.register_buffer("running_mean", torch.zeros(self.num_features))
            self.register_buffer("running_var", torch.ones(self.num_features))

        def forward(self, x):
            x = F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training,
                self.momentum,
                self.eps,
            )
            if self.activation == ACT_RELU:
                return F.relu(x, inplace=False)
            if self.activation == ACT_LEAKY_RELU:
                return F.leaky_relu(x, negative_slope=self.slope, inplace=False)
            if self.activation == ACT_ELU:
                return F.elu(x, inplace=False)
            return x

    InPlaceABNSync = ABN
    BatchNorm2d = functools.partial(InPlaceABNSync, activation=ACT_NONE)

    affine_par = True

    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            dilation=1,
            downsample=None,
            fist_dilation=1,  # noqa: ARG002 - kept for checkpoint compatibility
            multi_grid=1,
        ):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=dilation * multi_grid,
                dilation=dilation * multi_grid,
                bias=False,
            )
            self.bn2 = BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = BatchNorm2d(planes * 4)
            self.relu = nn.ReLU(inplace=False)
            self.relu_inplace = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.dilation = dilation
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out = out + residual
            out = self.relu_inplace(out)
            return out

    class PSPModule(nn.Module):
        def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
            super().__init__()
            self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
            self.bottleneck = nn.Sequential(
                nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, bias=False),
                InPlaceABNSync(out_features),
            )

        def _make_stage(self, features, out_features, size):
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
            conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
            bn = InPlaceABNSync(out_features)
            return nn.Sequential(prior, conv, bn)

        def forward(self, feats):
            h, w = feats.size(2), feats.size(3)
            priors = [
                F.interpolate(stage(feats), size=(h, w), mode="bilinear", align_corners=True) for stage in self.stages
            ] + [feats]
            bottle = self.bottleneck(torch.cat(priors, 1))
            return bottle

    class Edge_Module(nn.Module):
        def __init__(self, in_fea=(256, 512, 1024), mid_fea=256, out_fea=2):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, bias=False),
                InPlaceABNSync(mid_fea),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, bias=False),
                InPlaceABNSync(mid_fea),
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, bias=False),
                InPlaceABNSync(mid_fea),
            )
            self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, bias=True)
            self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, bias=True)

        def forward(self, x1, x2, x3):
            _, _, h, w = x1.size()
            edge1_fea = self.conv1(x1)
            edge1 = self.conv4(edge1_fea)
            edge2_fea = self.conv2(x2)
            edge2 = self.conv4(edge2_fea)
            edge3_fea = self.conv3(x3)
            edge3 = self.conv4(edge3_fea)

            edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode="bilinear", align_corners=True)
            edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode="bilinear", align_corners=True)
            edge2 = F.interpolate(edge2, size=(h, w), mode="bilinear", align_corners=True)
            edge3 = F.interpolate(edge3, size=(h, w), mode="bilinear", align_corners=True)

            edge = torch.cat([edge1, edge2, edge3], dim=1)
            edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
            edge = self.conv5(edge)
            return edge, edge_fea

    class Decoder_Module(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                InPlaceABNSync(256),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(256, 48, kernel_size=1, bias=False),
                InPlaceABNSync(48),
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(304, 256, kernel_size=1, bias=False),
                InPlaceABNSync(256),
                nn.Conv2d(256, 256, kernel_size=1, bias=False),
                InPlaceABNSync(256),
            )
            self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)

        def forward(self, xt, xl):
            _, _, h, w = xl.size()
            xt = F.interpolate(self.conv1(xt), size=(h, w), mode="bilinear", align_corners=True)
            xl = self.conv2(xl)
            x = torch.cat([xt, xl], dim=1)
            x = self.conv3(x)
            seg = self.conv4(x)
            return seg, x

    class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes: int):
            super().__init__()
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = BatchNorm2d(64)
            self.relu1 = nn.ReLU(inplace=False)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = BatchNorm2d(64)
            self.relu2 = nn.ReLU(inplace=False)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = BatchNorm2d(128)
            self.relu3 = nn.ReLU(inplace=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

            self.context_encoding = PSPModule(2048, 512)
            self.edge = Edge_Module()
            self.decoder = Decoder_Module(num_classes)
            self.fushion = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, bias=False),
                InPlaceABNSync(256),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1, bias=True),
            )

        def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    BatchNorm2d(planes * block.expansion, affine=affine_par),
                )

            layers = []
            if isinstance(multi_grid, tuple):
                grids = multi_grid

                def _mg(idx):
                    return grids[idx % len(grids)]

            else:

                def _mg(_idx):
                    return 1

            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    dilation=dilation,
                    downsample=downsample,
                    multi_grid=_mg(0),
                )
            )
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=_mg(i)))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.maxpool(x)
            x2 = self.layer1(x)
            x3 = self.layer2(x2)
            x4 = self.layer3(x3)
            x5 = self.layer4(x4)
            x = self.context_encoding(x5)
            parsing_result, parsing_fea = self.decoder(x, x2)
            edge_result, edge_fea = self.edge(x2, x3, x4)
            x = torch.cat([parsing_fea, edge_fea], dim=1)
            fusion_result = self.fushion(x)
            return [[parsing_result, fusion_result], [edge_result]]

    return ResNet(Bottleneck, [3, 4, 23, 3], int(num_classes))


def _load_checkpoint_state_dict(checkpoint_path: Path) -> dict[str, object]:
    torch, _nn, _F = _import_torch()
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError("Unsupported checkpoint format (expected dict or dict with 'state_dict').")

    fixed: dict[str, object] = {}
    for k, v in state.items():
        if not isinstance(k, str):
            continue
        key = k[7:] if k.startswith("module.") else k
        fixed[key] = v
    return fixed


def export_onnx(
    *,
    checkpoint: Path,
    dataset: Literal["lip", "atr", "pascal"],
    output: Path,
    opset: int,
    dynamic: bool,
    external_data: bool,
    allow_partial: bool,
    ir_version: int,
) -> None:
    torch, nn, _F = _import_torch()

    ds = DATASETS[dataset]
    opset = int(opset)
    if opset < 18:
        print(
            f"[warn] opset {opset} is likely to trigger exporter version-conversion; bumping to opset 18.",
            file=sys.stderr,
        )
        opset = 18
    model = build_schp_resnet101(num_classes=ds.num_classes)

    state = _load_checkpoint_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        msg = (
            f"Checkpoint mismatch for dataset={dataset}. "
            f"missing={len(missing)} unexpected={len(unexpected)}. "
            "Make sure you selected the correct dataset checkpoint (e.g. lip.pth with --dataset lip)."
        )
        if not allow_partial:
            raise RuntimeError(msg)
        print(f"[warn] {msg}", file=sys.stderr)
        if missing:
            print(f"[warn] missing keys (first 10): {missing[:10]}", file=sys.stderr)
        if unexpected:
            print(f"[warn] unexpected keys (first 10): {unexpected[:10]}", file=sys.stderr)

    import torch.nn.functional as F

    class FusionWrapper(nn.Module):
        def __init__(self, inner, out_hw: tuple[int, int]):
            super().__init__()
            self.inner = inner
            self.out_hw = out_hw

        def forward(self, x):
            out = self.inner(x)[0][1]
            # Export full-resolution logits, matching the SCHP extractor behavior.
            return F.interpolate(out, size=self.out_hw, mode="bilinear", align_corners=True)

    h, w = ds.input_size_hw
    wrapper = FusionWrapper(model, out_hw=(h, w)).eval()

    dummy = torch.randn(1, 3, h, w, dtype=torch.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[info] exporting {dataset} ({ds.num_classes} classes) to {output}")
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {"input": {0: "N", 2: "H", 3: "W"}, "logits": {0: "N", 2: "H", 3: "W"}}

    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    if external_data or int(ir_version) > 0:
        try:
            import onnx  # type: ignore

            try:
                model_onnx = onnx.load(str(output), load_external_data=True)
            except TypeError:
                model_onnx = onnx.load(str(output))

            if int(ir_version) > 0 and int(model_onnx.ir_version) > int(ir_version):
                print(
                    f"[info] lowering ONNX IR version {int(model_onnx.ir_version)} -> {int(ir_version)} for compatibility",
                    file=sys.stderr,
                )
                model_onnx.ir_version = int(ir_version)

            if external_data:
                try:
                    onnx.save_model(
                        model_onnx,
                        str(output),
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=output.name + ".data",
                        size_threshold=1024,
                        convert_attribute=False,
                    )
                except TypeError:
                    from onnx.external_data_helper import convert_model_to_external_data  # type: ignore

                    convert_model_to_external_data(
                        model_onnx,
                        all_tensors_to_one_file=True,
                        location=output.name + ".data",
                        size_threshold=1024,
                        convert_attribute=False,
                    )
                    onnx.save_model(model_onnx, str(output))
            else:
                onnx.save_model(model_onnx, str(output))
        except Exception as exc:
            print(f"[warn] could not post-process ONNX file (ir/external data): {exc}", file=sys.stderr)

    print("[info] done")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export SCHP (LIP/ATR/Pascal) checkpoint to ONNX (CPU-friendly).")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to SCHP checkpoint (.pth/.pth.tar). For example: lip.pth from HF.",
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), default="lip")
    parser.add_argument("--output", type=Path, default=Path("models/schp_lip.onnx"))
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--ir-version",
        type=int,
        default=9,
        help="Force ONNX IR version (default 9 for broad onnxruntime compatibility). Use 0 to leave as-is.",
    )
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic H/W axes (optional).")
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="Store weights in a separate .onnx.data file (recommended for large models).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow exporting even if checkpoint keys don't fully match the model (not recommended).",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint.expanduser().resolve()
    if not ckpt.exists():
        print(f"[error] checkpoint not found: {ckpt}", file=sys.stderr)
        return 2

    out = args.output
    if out.is_dir():
        out = out / f"schp_{args.dataset}.onnx"

    export_onnx(
        checkpoint=ckpt,
        dataset=args.dataset,
        output=out,
        opset=args.opset,
        dynamic=bool(args.dynamic),
        external_data=bool(args.external_data),
        allow_partial=bool(args.allow_partial),
        ir_version=int(args.ir_version),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
