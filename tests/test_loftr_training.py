from pathlib import Path
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.training.loss import LoFTRLoss
from src.features.deep_matcher import DeepCorrespondenceMatcher

def test_loftr_loss_autograd_gradients():
    import kornia.feature as kf
    try:
        model = kf.LoFTR(pretrained=None)
    except Exception:
        pytest.skip("Kornia LoFTR could not be instantiated")

    loss_fn = LoFTRLoss(coarse_weight=1.0, fine_weight=0.25, temperature=1.0)
    
    # 64x64 dummy input (divisible by 8)
    img0 = torch.rand(1, 1, 64, 64, requires_grad=False)
    img1 = torch.rand(1, 1, 64, 64, requires_grad=False)
    
    # 5 dummy matching points
    k0 = torch.tensor([[10.0, 12.0], [20.0, 25.0], [30.0, 32.0], [40.0, 42.0], [50.0, 52.0]])
    k1 = torch.tensor([[12.0, 14.0], [22.0, 27.0], [32.0, 34.0], [42.0, 44.0], [52.0, 54.0]])

    total_loss, loss_dict = loss_fn.forward_train_step(model, img0, img1, k0, k1)
    
    assert torch.is_tensor(total_loss)
    assert total_loss.item() > 0.0
    assert "loss_coarse" in loss_dict

    # Verify backward autograd pass
    model.zero_grad()
    total_loss.backward()

    # Check that coarse transformer and backbone parameters receive non-zero gradients
    has_coarse_grads = any(p.grad is not None and torch.norm(p.grad).item() > 0 for p in model.loftr_coarse.parameters())
    has_backbone_grads = any(p.grad is not None and torch.norm(p.grad).item() > 0 for p in model.backbone.parameters())
    assert has_coarse_grads, "LoFTR coarse transformer must receive gradients from LoFTRLoss"
    assert has_backbone_grads, "Backbone parameters must receive gradients from LoFTRLoss"

def test_checkpoint_loading_and_fallback(monkeypatch):
    pytest.importorskip("kornia")
    import kornia.feature as kf

    # 1. Existing lunar fine-tuned checkpoint
    ckpt_path = Path("models/loftr/lunar_finetuned/best.ckpt")
    if ckpt_path.exists():
        matcher = DeepCorrespondenceMatcher(backend="LOFTR", checkpoint_path=str(ckpt_path))
        assert matcher.loftr_mode == "lunar_finetuned"
        assert matcher.checkpoint_loaded is True
        assert "Lunar Fine-Tuned" in matcher.active_backend

    # 2. When checkpoint is absent, fallback to pretrained LoFTR (offline mock)
    class MockPretrainedLoFTR(torch.nn.Module):
        def __init__(self, pretrained=None):
            super().__init__()
            self.pretrained = pretrained
        def forward(self, x):
            return {"keypoints0": torch.empty((0, 2)), "keypoints1": torch.empty((0, 2)), "confidence": torch.empty((0,))}

    monkeypatch.setattr(kf, "LoFTR", MockPretrainedLoFTR)
    fallback_pretrained = DeepCorrespondenceMatcher(backend="LOFTR", checkpoint_path="non_existent/path.ckpt")
    assert fallback_pretrained.checkpoint_loaded is False
    assert fallback_pretrained.loftr_mode == "pretrained"
    assert "Pretrained Outdoor Fallback" in fallback_pretrained.active_backend

    # 3. When pretrained LoFTR fails (e.g. offline network failure), fallback to CNN
    def mock_loftr_network_failure(*args, **kwargs):
        raise OSError("Offline: simulated network failure downloading weights")

    monkeypatch.setattr(kf, "LoFTR", mock_loftr_network_failure)
    fallback_cnn = DeepCorrespondenceMatcher(backend="LOFTR", checkpoint_path="non_existent/path.ckpt")
    assert fallback_cnn.checkpoint_loaded is False
    assert fallback_cnn.loftr_mode == "fallback_cnn"
    assert "FALLBACK_LUNARNET" in fallback_cnn.active_backend
