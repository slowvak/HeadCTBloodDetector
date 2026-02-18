"""MLX re-implementation of DeepMedic for Apple Silicon inference.

Layout convention: NDHWC (channels-last) throughout, matching MLX's Conv3d.
"""
import mlx.core as mx
import mlx.nn as nn

SCALE_FACTORS = ((5, 5, 5), (3, 3, 3), (1, 1, 1))
FEATURE_MAPS = (30, 30, 40, 40, 40, 40, 50, 50)
FULLY_CONNECTED = (250, 250)
DROPOUT = (0.0, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Size-calculation helpers (same maths as base.py, no torch dependency)
# ---------------------------------------------------------------------------

def _conv_out(input_size, kernel_size, padding, stride):
    return tuple(
        (i + 2 * p - k) // s + 1
        for i, p, k, s in zip(input_size, padding, kernel_size, stride)
    )


def _conv_in(output_size, kernel_size, padding, stride):
    return tuple(
        (o - 1) * s + k - 2 * p
        for o, p, k, s in zip(output_size, padding, kernel_size, stride)
    )


def crop_center(x: mx.array, size: tuple) -> mx.array:
    """Center-crop spatial dims (D, H, W) of an NDHWC array."""
    if tuple(x.shape[1:4]) == size:
        return x
    slices = [slice(None)]  # N
    for c, s in zip(x.shape[1:4], size):
        start = c // 2 - s // 2
        slices.append(slice(start, start + s))
    slices.append(slice(None))  # C
    return x[tuple(slices)]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DownSample(nn.Module):
    """Stride-based downsample along spatial dims — no learnable parameters."""

    def __init__(self, scale_factor: tuple):
        super().__init__()
        self.scale_factor = tuple(scale_factor)
        # NDHWC: batch=0, D=1, H=2, W=3, C=4
        self._slices = (slice(None),) + \
                       tuple(slice(s // 2, None, s) for s in self.scale_factor) + \
                       (slice(None),)

    def __call__(self, x: mx.array) -> mx.array:
        return x[self._slices]

    def calculate_input_size(self, output_size):
        return tuple(o * s for o, s in zip(output_size, self.scale_factor))

    def calculate_output_size(self, input_size):
        return tuple((i - s) // s + 1 for i, s in zip(input_size, self.scale_factor))

    def update_fov_and_scale_factor(self, fov, scale_factor):
        scale_factor = tuple(s0 * s1 for s0, s1 in zip(scale_factor, self.scale_factor))
        return fov, scale_factor


class UpSample(nn.Module):
    """Nearest-neighbour upsample via mx.repeat — no learnable parameters."""

    def __init__(self, scale_factor: tuple):
        super().__init__()
        self.scale_factor = tuple(scale_factor)

    def __call__(self, x: mx.array) -> mx.array:
        # Repeat each voxel s times along spatial axes D(1), H(2), W(3)
        for axis, s in enumerate(self.scale_factor, start=1):
            x = mx.repeat(x, s, axis=axis)
        return x

    def calculate_input_size(self, output_size):
        return tuple(o // s + bool(o % s) for o, s in zip(output_size, self.scale_factor))

    def calculate_output_size(self, input_size):
        return tuple(i * s for i, s in zip(input_size, self.scale_factor))

    def update_fov_and_scale_factor(self, fov, scale_factor):
        scale_factor = tuple(s0 / float(s1) for s0, s1 in zip(scale_factor, self.scale_factor))
        return fov, scale_factor


class PReLU(nn.Module):
    """Parametric ReLU with a single learnable slope (matches PyTorch naming)."""

    def __init__(self):
        super().__init__()
        self.weight = mx.array([0.25])

    def __call__(self, x: mx.array) -> mx.array:
        return mx.maximum(0.0, x) + self.weight * mx.minimum(0.0, x)


class PreActBlock(nn.Module):
    """Pre-activation residual block: BN → PReLU → Dropout → Conv3d.

    Mirrors blast_ct/models/base.py PreActBlock, NDHWC layout.
    """

    def __init__(self, in_planes: int, planes: int,
                 kernel_size=(3, 3, 3), stride=(1, 1, 1), dropout_prob: float = 0.0):
        super().__init__()
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self._padding = (0, 0, 0)

        self.bn = nn.BatchNorm(in_planes)
        self.prelu = PReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        # MLX Conv3d: weight shape (C_out, kD, kH, kW, C_in)
        self.conv = nn.Conv3d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
        )
        # Shortcut slices in NDHWC layout: (batch, D, H, W, C)
        # Crop spatial dims to match conv output; keep all channels.
        self._shortcut_slices = (slice(None),) + tuple(
            slice(k // 2, None if k == 1 else -(k // 2), s)
            for k, s in zip(kernel_size, stride)
        ) + (slice(None),)

    def __call__(self, x: mx.array) -> mx.array:
        # MLX BatchNorm only handles ≤4-D tensors; merge N and D temporarily.
        n, d, h, w, c = x.shape
        out = self.bn(x.reshape(n * d, h, w, c)).reshape(n, d, h, w, c)
        out = self.prelu(out)
        out = self.dropout(out)
        out = self.conv(out)
        # Skip connection when channel count is unchanged
        if x.shape[-1] == out.shape[-1]:
            out = out + x[self._shortcut_slices]
        return out

    def calculate_input_size(self, output_size):
        return _conv_in(output_size, self.kernel_size, self._padding, self.stride)

    def calculate_output_size(self, input_size):
        return _conv_out(input_size, self.kernel_size, self._padding, self.stride)

    def update_fov_and_scale_factor(self, fov, scale_factor):
        fov = tuple(f + s * (k - 1) for f, s, k in zip(fov, scale_factor, self.kernel_size))
        return fov, scale_factor


# ---------------------------------------------------------------------------
# Path and top-level model
# ---------------------------------------------------------------------------

class Path(nn.Module):
    """One multi-resolution path: downsample → conv blocks → upsample."""

    def __init__(self, scale_factor, input_channels: int, feature_maps):
        super().__init__()
        layers = [DownSample(tuple(scale_factor))]
        for i, fm in enumerate(feature_maps):
            in_ch = feature_maps[i - 1] if i > 0 else input_channels
            layers.append(PreActBlock(in_ch, fm))
        layers.append(UpSample(tuple(scale_factor)))
        # MLX Sequential stores layers in .layers — matches the key mapping
        self.path = nn.Sequential(*layers)

    @property
    def _size_layers(self):
        """Non-stored view of layers for size calculation (avoids duplicate MLX tracking)."""
        return self.path.layers

    def calculate_input_size(self, output_size):
        size = output_size
        for layer in reversed(self._size_layers):
            size = layer.calculate_input_size(size)
        return size

    def calculate_output_size(self, input_size):
        size = input_size
        for layer in self._size_layers:
            size = layer.calculate_output_size(size)
        return size

    def __call__(self, x: mx.array, output_size: tuple) -> mx.array:
        input_size = self.calculate_input_size(output_size)
        out = crop_center(x, input_size)
        out = self.path(out)
        out = crop_center(out, output_size)
        return out


class DeepMedic(nn.Module):
    """MLX DeepMedic: three-path 3-D CNN for brain lesion segmentation."""

    def __init__(self, input_channels: int, num_classes: int,
                 scale_factors=SCALE_FACTORS, feature_maps=FEATURE_MAPS,
                 fully_connected=FULLY_CONNECTED, dropout=DROPOUT):
        super().__init__()
        assert len(dropout) == len(fully_connected) + 1

        # Stored as a list → MLX tree keys: paths.0.*, paths.1.*, paths.2.*
        self.paths = [Path(sf, input_channels, feature_maps) for sf in scale_factors]

        channels = (feature_maps[-1] * len(self.paths),) + tuple(fully_connected) + (num_classes,)
        self.fully_connected = nn.Sequential(*[
            PreActBlock(channels[i], channels[i + 1], kernel_size=(1, 1, 1), dropout_prob=dropout[i])
            for i in range(len(channels) - 1)
        ])
        # Plain Python dict — not tracked by MLX (no mx.array values)
        self._output_size_cache: dict = {}

    def get_output_size(self, input_size: tuple) -> tuple:
        if input_size not in self._output_size_cache:
            self._output_size_cache[input_size] = self.paths[0].calculate_output_size(input_size)
        return self._output_size_cache[input_size]

    def __call__(self, image: mx.array, **kwargs):
        # image: (N, D, H, W, C)
        input_size = tuple(image.shape[1:4])
        output_size = self.get_output_size(input_size)
        activations = [path(image, output_size) for path in self.paths]
        out = mx.concatenate(activations, axis=-1)  # concat along channel axis
        out = self.fully_connected(out)
        return out, {}
