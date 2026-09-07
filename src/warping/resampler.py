"""
Scientific Image Warping and Resampling Layer.
Preserves reference coordinate bounds and pixel resolution with high-order interpolation.
"""
from typing import Tuple, Optional
import numpy as np
import cv2
from src.geometry.models import GeometricModel, HomographyModel, AffineModel, SimilarityModel

class ImageWarper:
    def __init__(self, interpolation: str = "bilinear"):
        self.interpolation = interpolation.lower()
        if self.interpolation in ["bicubic", "cubic"]:
            self.cv_interp = cv2.INTER_CUBIC
        elif self.interpolation == "nearest":
            self.cv_interp = cv2.INTER_NEAREST
        else:
            self.cv_interp = cv2.INTER_LINEAR

    def warp(
        self,
        source_img: np.ndarray,
        model: GeometricModel,
        reference_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp source image into reference coordinate grid.
        Returns:
            warped_image: ndarray matching reference_shape
            valid_mask: boolean mask of transformed valid data
        """
        ref_h, ref_w = reference_shape[:2]

        is_float = source_img.dtype in [np.float32, np.float64]
        src_f = source_img.astype(np.float32)

        if isinstance(model, HomographyModel):
            warped = cv2.warpPerspective(
                src_f,
                model.matrix,
                (ref_w, ref_h),
                flags=self.cv_interp,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0
            )
            # Mask
            ones_mask = np.ones(source_img.shape[:2], dtype=np.uint8)
            valid_mask_8u = cv2.warpPerspective(
                ones_mask,
                model.matrix,
                (ref_w, ref_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        elif isinstance(model, (AffineModel, SimilarityModel)):
            warped = cv2.warpAffine(
                src_f,
                model.matrix,
                (ref_w, ref_h),
                flags=self.cv_interp,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0
            )
            ones_mask = np.ones(source_img.shape[:2], dtype=np.uint8)
            valid_mask_8u = cv2.warpAffine(
                ones_mask,
                model.matrix,
                (ref_w, ref_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            raise TypeError(f"Unsupported model type for warping: {type(model)}")

        valid_mask = valid_mask_8u.astype(bool)

        if not is_float and source_img.dtype == np.uint8:
            warped = np.clip(warped, 0, 255).astype(np.uint8)

        return warped, valid_mask
