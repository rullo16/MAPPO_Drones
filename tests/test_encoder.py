import pytest
import torch

from MAPPO.vision_encoders import EfficientVisionEncoder


@pytest.mark.parametrize("input_shape,output_dim", [
    ((4, 84, 84), 256),   # default: conv-native dim, Identity projection
    ((4, 84, 84), 128),   # projection engaged
    ((1, 36, 36), 256),
    ((3, 64, 64), 512),   # historically crashed: Linear(512,...) fed 256 dims
])
def test_output_dim_matches(input_shape, output_dim):
    enc = EfficientVisionEncoder(input_shape, output_dim=output_dim)
    out = enc(torch.zeros(2, *input_shape))
    assert out.shape == (2, output_dim)
    assert enc.output_dim == output_dim


def test_identity_projection_for_native_dim():
    enc = EfficientVisionEncoder((4, 84, 84), output_dim=256)
    assert isinstance(enc.projection, torch.nn.Identity)
    assert enc.conv_dim == 256
