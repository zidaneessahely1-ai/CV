"""Tests for image processing utilities."""

import numpy as np

from src.utils.image_processing import draw_detections, image_to_bytes, load_image


class TestLoadImage:
    def test_load_from_bytes(self):
        """Load a valid image from raw bytes."""
        from io import BytesIO

        from PIL import Image

        buf = BytesIO()
        Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(buf, "PNG")
        img = load_image(buf.getvalue())
        assert img.shape == (50, 50, 3)

    def test_load_invalid_path(self):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_image("/tmp/nonexistent_abc123.png")


class TestDrawDetections:
    def test_draw_returns_copy(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        dets = [
            {"bbox": [10, 10, 50, 50], "class_name": "person", "confidence": 0.95}
        ]
        result = draw_detections(img, dets)
        assert result is not img
        assert result.shape == img.shape

    def test_draw_empty_detections(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = draw_detections(img, [])
        assert np.array_equal(result, img)


class TestImageToBytes:
    def test_encode_jpeg(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        data = image_to_bytes(img, ".jpg")
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_encode_png(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        data = image_to_bytes(img, ".png")
        assert isinstance(data, bytes)
        assert len(data) > 0
