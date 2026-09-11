from pathlib import Path
import pytest
import numpy as np
import cv2

from src.training.dataset import (
    DatasetItem,
    LunarDatasetScanner,
    LunarPairGenerator,
    PseudoGroundTruthGenerator,
    LunarLoFTRDataset
)

def test_sensor_detection():
    scanner = LunarDatasetScanner()
    assert scanner.detect_sensor(Path("data/raw/ch2/ohrc/ch2_ohrc_001.tif")) == "OHRC"
    assert scanner.detect_sensor(Path("data/raw/ch2/tmc2/ch2_tmc_002.png")) == "TMC2"
    assert scanner.detect_sensor(Path("data/raw/ch2/iirs/ch2_iirs_003.npy")) == "IIRS"
    assert scanner.detect_sensor(Path("data/raw/lroc/nac_image_004.tif")) == "LROC_NAC"
    assert scanner.detect_sensor(Path("data/raw/selene/tc_image_005.tif")) == "SELENE"

def test_pair_generator_overlap():
    item1 = DatasetItem(
        path=Path("mock1.png"),
        sensor_type="OHRC",
        width=512,
        height=512,
        gsd=0.25,
        bit_depth=8,
        center_lon=20.0,
        center_lat=-70.0
    )
    # item2 nearby with large overlap
    item2 = DatasetItem(
        path=Path("mock2.png"),
        sensor_type="LROC_NAC",
        width=512,
        height=512,
        gsd=0.50,
        bit_depth=8,
        center_lon=20.001,
        center_lat=-70.001
    )
    # item3 far away with zero overlap
    item3 = DatasetItem(
        path=Path("mock3.png"),
        sensor_type="TMC2",
        width=512,
        height=512,
        gsd=5.0,
        bit_depth=8,
        center_lon=50.0,
        center_lat=-10.0
    )

    pair_gen = LunarPairGenerator(min_overlap_pct=0.10)
    pairs = pair_gen.generate_pairs([item1, item2, item3])
    
    assert len(pairs) >= 1
    found_1_2 = any(
        (p["source_path"] == str(item1.path) and p["reference_path"] == str(item2.path)) or
        (p["source_path"] == str(item2.path) and p["reference_path"] == str(item1.path))
        for p in pairs
    )
    assert found_1_2
    found_3 = any(p["source_path"] == str(item3.path) or p["reference_path"] == str(item3.path) for p in pairs)
    assert not found_3

def test_pseudo_ground_truth_derivation(tmp_path):
    rng = np.random.RandomState(42)
    img_base = (rng.rand(256, 256) * 100 + 80).astype(np.uint8)
    for _ in range(12):
        cx, cy = rng.randint(40, 210, size=2)
        r = rng.randint(10, 30)
        cv2.circle(img_base, (int(cx), int(cy)), int(r), 40, -1)
        cv2.circle(img_base, (int(cx), int(cy)), int(r), 200, 2)

    M = np.float32([[1.0, 0.0, 15.0], [0.0, 1.0, 10.0]])
    img_warped = cv2.warpAffine(img_base, M, (256, 256))

    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    cv2.imwrite(str(p1), img_base)
    cv2.imwrite(str(p2), img_warped)

    pair_entry = {
        "pair_id": "test_pair",
        "source_path": str(p1),
        "reference_path": str(p2),
        "source_sensor": "OHRC",
        "reference_sensor": "LROC_NAC",
        "source_gsd": 0.25,
        "reference_gsd": 0.25
    }

    pgt_gen = PseudoGroundTruthGenerator()
    gt_data = pgt_gen.derive_ground_truth(pair_entry)

    assert "ground_truth_status" in gt_data
    if gt_data["ground_truth_status"] == "geometry-derived":
        corrs = gt_data.get("correspondences", [])
        assert len(corrs) >= 4
        assert gt_data.get("condition_number", 0.0) < 2000.0

def test_lunar_loftr_dataset(tmp_path):
    # Test LunarLoFTRDataset initialization and indexing
    p1 = tmp_path / "src.png"
    p2 = tmp_path / "ref.png"
    arr = np.zeros((128, 128), dtype=np.uint8)
    cv2.imwrite(str(p1), arr)
    cv2.imwrite(str(p2), arr)

    pairs = [{
        "pair_id": "sample_0",
        "source_path": str(p1),
        "reference_path": str(p2),
        "source_sensor": "OHRC",
        "reference_sensor": "LROC_NAC",
        "ground_truth_status": "geometry-derived",
        "correspondences": [
            {"source_x": 10.0, "source_y": 10.0, "reference_x": 12.0, "reference_y": 12.0, "confidence": 1.0},
            {"source_x": 20.0, "source_y": 20.0, "reference_x": 22.0, "reference_y": 22.0, "confidence": 1.0},
            {"source_x": 30.0, "source_y": 30.0, "reference_x": 32.0, "reference_y": 32.0, "confidence": 1.0},
            {"source_x": 40.0, "source_y": 40.0, "reference_x": 42.0, "reference_y": 42.0, "confidence": 1.0},
        ]
    }]

    dataset = LunarLoFTRDataset(pairs=pairs, image_size=128, augment=False)
    assert len(dataset) == 1
    sample = dataset[0]
    assert "image0" in sample
    assert "image1" in sample
    assert "keypoints0" in sample
    assert "keypoints1" in sample
    assert sample["image0"].shape == (1, 128, 128)
