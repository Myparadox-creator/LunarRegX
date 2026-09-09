from pathlib import Path
import pytest
import numpy as np
from src.multimodal.sensor_encoder import (
    OHRCEncoder,
    TMC2Encoder,
    IIRSEncoder,
    encode_sensor_image,
    CommonTerrainRepresentation
)
from src.illumination.augmentation import LunarPhysicsAugmenter
from src.io.dataset import LunarImage, SensorMetadata

def test_ohrc_encoder():
    raw = (np.random.rand(100, 100) * 255).astype(np.uint8)
    meta = SensorMetadata(sensor_name="CHANDRAYAAN2_OHRC", gsd=0.25)
    img = LunarImage(raw_array=raw, normalized=raw/255.0, display_8bit=raw, metadata=meta)
    
    rep = OHRCEncoder().encode(img)
    assert isinstance(rep, CommonTerrainRepresentation)
    assert rep.sensor_type == "OHRC"
    assert rep.structural_map.shape == (100, 100)
    assert rep.effective_gsd_m == 0.25

def test_tmc2_encoder():
    raw = (np.random.rand(100, 100) * 255).astype(np.uint8)
    meta = SensorMetadata(sensor_name="CHANDRAYAAN2_TMC2", gsd=5.0)
    img = LunarImage(raw_array=raw, normalized=raw/255.0, display_8bit=raw, metadata=meta)
    
    rep = TMC2Encoder().encode(img)
    assert rep.sensor_type == "TMC2"
    assert rep.structural_map.shape == (100, 100)
    assert rep.effective_gsd_m == 5.0

def test_iirs_hyperspectral_encoder():
    # 3D Hyperspectral cube: (H, W, Bands) e.g. 50x50 with 16 spectral channels
    cube = (np.random.rand(50, 50, 16) * 1000.0).astype(np.float32)
    meta = SensorMetadata(sensor_name="CHANDRAYAAN2_IIRS", gsd=80.0)
    norm = cube[:, :, 0] / 1000.0
    img = LunarImage(raw_array=cube, normalized=norm, display_8bit=(norm*255).astype(np.uint8), metadata=meta)
    
    rep = IIRSEncoder().encode(img)
    assert rep.sensor_type == "IIRS"
    assert rep.structural_map.shape == (50, 50)
    assert rep.spectral_channels == 16
    assert rep.effective_gsd_m == 80.0

def test_sensor_dispatch_factory():
    raw = np.ones((32, 32), dtype=np.uint8) * 128
    img_ohrc = LunarImage(raw_array=raw, normalized=raw/255.0, display_8bit=raw, metadata=SensorMetadata(sensor_name="OHRC", gsd=0.25))
    img_iirs = LunarImage(raw_array=raw, normalized=raw/255.0, display_8bit=raw, metadata=SensorMetadata(sensor_name="IIRS", gsd=80.0))
    
    rep_ohrc = encode_sensor_image(img_ohrc)
    rep_iirs = encode_sensor_image(img_iirs)
    assert rep_ohrc.sensor_type == "OHRC"
    assert rep_iirs.sensor_type == "IIRS"

def test_lunar_physics_augmentation():
    img_norm = np.ones((64, 64), dtype=np.float32) * 0.5
    augmenter = LunarPhysicsAugmenter(seed=123)
    aug_img, meta = augmenter.augment(img_norm)
    assert aug_img.shape == (64, 64)
    assert "applied_augmentations" in meta
    assert len(meta["applied_augmentations"]) > 0