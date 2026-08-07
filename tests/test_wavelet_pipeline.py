import json

import numpy as np
import pywt
import torch

from WSGM_Foward_and_Generator import inverse_dwt_batch
from lib.diffusion_lib.UNet import UNet


def test_periodic_wavelet_roundtrip_is_exact():
    image = np.random.default_rng(0).normal(size=(64, 64)).astype(np.float32)
    for level in (1, 2, 3):
        coeffs = pywt.wavedec2(image, "db1", mode="periodization", level=level)
        reconstructed = pywt.waverec2(coeffs, "db1", mode="periodization")
        np.testing.assert_allclose(reconstructed, image, rtol=1e-6, atol=1e-6)


def test_inverse_dwt_batch_has_expected_shape():
    ca = torch.randn(3, 1, 8, 8)
    details = torch.randn(3, 3, 8, 8)
    image = inverse_dwt_batch(ca, details, wavelet="db1", mode="periodization")
    assert image.shape == (3, 1, 16, 16)
    assert torch.isfinite(image).all()


def test_stationary_unet_variant_forward_and_structure():
    model = UNet(
        size=16,
        in_channels=4,
        out_channels=3,
        depth=2,
        blocks_per_level=1,
        base_channels=8,
        continuous_time=True,
        norm_type="group",
        upsample_mode="interpolate",
    )
    x = torch.randn(2, 4, 16, 16)
    t = torch.tensor([0.2, 0.8])
    y = model(x, t)
    assert y.shape == (2, 3, 16, 16)
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in model.modules())
    assert any(isinstance(module, torch.nn.Upsample) for module in model.modules())


def test_legacy_unet_state_dict_names_are_preserved():
    model = UNet(size=16, in_channels=1, out_channels=1, depth=1, blocks_per_level=1)
    assert any(name.endswith(".ln.weight") for name in model.state_dict())


def test_active_wavelet_config_declares_new_ablation_variant():
    with open("WSGM_Config.json", encoding="utf-8") as file:
        config = json.load(file)
    assert config["model"]["norm_type"] in {"layer", "group"}
    assert config["model"]["upsample_mode"] in {"conv_transpose", "interpolate"}

