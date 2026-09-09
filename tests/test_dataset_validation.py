from pathlib import Path
import tempfile
import pytest
import numpy as np
from PIL import Image

from src.io.dataset import load_lunar_image

def test_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_lunar_image('non_existent_file_path.tif')

def test_empty_file_raises_error(tmp_path):
    empty_file = tmp_path / 'empty.tif'
    empty_file.write_bytes(b'')
    with pytest.raises(ValueError, match='0 bytes'):
        load_lunar_image(empty_file)

def test_corrupt_file_raises_error(tmp_path):
    corrupt_file = tmp_path / 'corrupt.png'
    corrupt_file.write_bytes(b'Not an image file content header')
    with pytest.raises(ValueError, match='Failed to decode'):
        load_lunar_image(corrupt_file)

def test_tiny_image_raises_error(tmp_path):
    tiny_file = tmp_path / 'tiny.png'
    tiny_arr = np.zeros((4, 4), dtype=np.uint8)
    Image.fromarray(tiny_arr).save(tiny_file)
    with pytest.raises(ValueError, match='too small'):
        load_lunar_image(tiny_file)

def test_valid_image_loads_successfully(tmp_path):
    valid_file = tmp_path / 'valid.png'
    arr = (np.random.rand(64, 64) * 255).astype(np.uint8)
    Image.fromarray(arr).save(valid_file)
    lunar_img = load_lunar_image(valid_file, gsd_override=0.25)
    assert lunar_img.width == 64
    assert lunar_img.height == 64
    assert lunar_img.metadata.gsd == 0.25
    assert lunar_img.display_8bit.shape == (64, 64)
