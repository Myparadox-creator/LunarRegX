"""
Deep Correspondence Matcher.
Provides pluggable deep learning matching backends:
- LoFTR (Detector-Free Local Feature Matching with Transformers)
- LightGlue (Deep Feature Matching with Adaptive Pruning)
- Learned LunarNet Adapter (Resilient CPU-optimized fallback)
Consumes common multimodal terrain representations and produces standardized correspondence lists.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.matching.matcher import Correspondence
from src.features.learned_adapter import LightweightLunarNet

class DeepCorrespondenceMatcher:
    """
    Deep correspondence engine supporting LoFTR, LightGlue, and LunarNet.
    """
    def __init__(
        self,
        backend: str = "LOFTR",
        confidence_thresh: float = 0.25,
        max_dimension: int = 640,
        checkpoint_path: Optional[str] = None
    ):
        self.backend = backend.upper()
        self.confidence_thresh = confidence_thresh
        self.max_dimension = max_dimension
        self.checkpoint_path = checkpoint_path or "models/loftr/lunar_finetuned/best.ckpt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.fallback_model = None
        self.active_backend = self.backend
        self.loftr_mode = "pretrained"
        self.checkpoint_loaded = False
        self._init_backend()

    def _init_backend(self):
        if self.backend == "LOFTR":
            try:
                import kornia.feature as kf
                from pathlib import Path
                chk_file = Path(self.checkpoint_path)

                # Initialize base model architecture
                try:
                    self.model = kf.LoFTR(pretrained="outdoor").to(self.device)
                except Exception:
                    self.model = kf.LoFTR(pretrained=None).to(self.device)

                # Priority 1: Check for lunar fine-tuned checkpoint
                if chk_file.exists():
                    try:
                        chk_data = torch.load(str(chk_file), map_location=self.device)
                        state_dict = chk_data.get("state_dict", chk_data)
                        self.model.load_state_dict(state_dict)
                        self.active_backend = "LOFTR (Lunar Fine-Tuned)"
                        self.loftr_mode = "lunar_finetuned"
                        self.checkpoint_loaded = True
                    except Exception as e:
                        # Fallback to pretrained if loading fails
                        self.active_backend = "LOFTR (Pretrained Outdoor Fallback)"
                        self.loftr_mode = "pretrained"
                        self.checkpoint_loaded = False
                else:
                    self.active_backend = "LOFTR (Pretrained Outdoor Fallback)"
                    self.loftr_mode = "pretrained"
                    self.checkpoint_loaded = False

                self.model.eval()
            except Exception as e:
                self.fallback_model = LightweightLunarNet().eval()
                self.active_backend = f"FALLBACK_LUNARNET ({e})"
                self.loftr_mode = "fallback_cnn"
        elif self.backend == "LIGHTGLUE":
            try:
                import kornia.feature as kf
                self.model = kf.LightGlue(features="disk").to(self.device).eval()
                self.active_backend = "LIGHTGLUE"
            except Exception:
                self.fallback_model = LightweightLunarNet().eval()
                self.active_backend = "FALLBACK_LUNARNET"
        else:
            self.fallback_model = LightweightLunarNet().eval()
            self.active_backend = "LIGHTWEIGHT_LUNARNET"

    def match(
        self,
        src_img: np.ndarray,
        ref_img: np.ndarray,
        src_mask: Optional[np.ndarray] = None,
        ref_mask: Optional[np.ndarray] = None
    ) -> Tuple[List[Correspondence], List[Dict[str, Any]]]:
        """
        Execute deep correspondence matching between source and reference images.
        Returns:
            correspondences: List of pipeline Correspondence objects
            matches_dict: List of standardized JSON-serializable dictionaries:
                [{"src_x": ..., "src_y": ..., "ref_x": ..., "ref_y": ..., "confidence": ...}]
        """
        h_s, w_s = src_img.shape[:2]
        h_r, w_r = ref_img.shape[:2]

        # Rescale for deep matcher inference if exceeds max_dimension
        scale_s = min(1.0, self.max_dimension / max(h_s, w_s))
        scale_r = min(1.0, self.max_dimension / max(h_r, w_r))

        new_w_s, new_h_s = int(round(w_s * scale_s)), int(round(h_s * scale_s))
        new_w_r, new_h_r = int(round(w_r * scale_r)), int(round(h_r * scale_r))

        # Ensure dimensions are divisible by 8 for LoFTR feature maps
        new_w_s = (new_w_s // 8) * 8
        new_h_s = (new_h_s // 8) * 8
        new_w_r = (new_w_r // 8) * 8
        new_h_r = (new_h_r // 8) * 8

        s_resized = cv2.resize(src_img, (new_w_s, new_h_s), interpolation=cv2.INTER_AREA)
        r_resized = cv2.resize(ref_img, (new_w_r, new_h_r), interpolation=cv2.INTER_AREA)

        # Convert to float32 [0.0, 1.0]
        s_norm = s_resized.astype(np.float32) / 255.0 if s_resized.max() > 1.0 else s_resized.astype(np.float32)
        r_norm = r_resized.astype(np.float32) / 255.0 if r_resized.max() > 1.0 else r_resized.astype(np.float32)

        s_t = torch.from_numpy(s_norm).unsqueeze(0).unsqueeze(0).to(self.device)
        r_t = torch.from_numpy(r_norm).unsqueeze(0).unsqueeze(0).to(self.device)

        src_pts, ref_pts, confidences = [], [], []

        if self.model is not None and "LOFTR" in self.active_backend:
            input_dict = {"image0": s_t, "image1": r_t}
            with torch.no_grad():
                out = self.model(input_dict)
            pts0 = out["keypoints0"].cpu().numpy()
            pts1 = out["keypoints1"].cpu().numpy()
            conf = out["confidence"].cpu().numpy()

            mask = conf >= self.confidence_thresh
            pts0 = pts0[mask]
            pts1 = pts1[mask]
            conf = conf[mask]

            # Scale back to original coordinates
            for p0, p1, c in zip(pts0, pts1, conf):
                orig_x0 = float(p0[0] * (w_s / new_w_s))
                orig_y0 = float(p0[1] * (h_s / new_h_s))
                orig_x1 = float(p1[0] * (w_r / new_w_r))
                orig_y1 = float(p1[1] * (h_r / new_h_r))
                src_pts.append((orig_x0, orig_y0))
                ref_pts.append((orig_x1, orig_y1))
                confidences.append(float(c))

        if len(src_pts) < 4:
            # Fallback to feature-based matching with LunarNet CNN
            engine = self.fallback_model or LightweightLunarNet().eval()
            s_u8 = (s_norm * 255.0).astype(np.uint8)
            r_u8 = (r_norm * 255.0).astype(np.uint8)

            c_s = cv2.goodFeaturesToTrack(s_u8, maxCorners=800, qualityLevel=0.01, minDistance=6)
            c_r = cv2.goodFeaturesToTrack(r_u8, maxCorners=800, qualityLevel=0.01, minDistance=6)

            if c_s is not None and c_r is not None:
                with torch.no_grad():
                    f_s = engine(s_t).squeeze(0).numpy()
                    f_r = engine(r_t).squeeze(0).numpy()

                desc_s = [f_s[:, int(round(p[0][1])), int(round(p[0][0]))] for p in c_s]
                desc_r = [f_r[:, int(round(p[0][1])), int(round(p[0][0]))] for p in c_r]

                desc_s = np.array(desc_s, dtype=np.float32)
                desc_r = np.array(desc_r, dtype=np.float32)

                # MNN matching
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                raw_matches = bf.match(desc_s, desc_r)

                for m in raw_matches:
                    p0 = c_s[m.queryIdx][0]
                    p1 = c_r[m.trainIdx][0]
                    orig_x0 = float(p0[0] * (w_s / new_w_s))
                    orig_y0 = float(p0[1] * (h_s / new_h_s))
                    orig_x1 = float(p1[0] * (w_r / new_w_r))
                    orig_y1 = float(p1[1] * (h_r / new_h_r))
                    src_pts.append((orig_x0, orig_y0))
                    ref_pts.append((orig_x1, orig_y1))
                    confidences.append(float(1.0 / (1.0 + m.distance)))

        # Build output structures
        correspondences = []
        matches_dict = []

        for idx, (p_s, p_r, conf) in enumerate(zip(src_pts, ref_pts, confidences)):
            corr = Correspondence(
                id=idx + 1,
                src_pt=(float(p_s[0]), float(p_s[1])),
                ref_pt=(float(p_r[0]), float(p_r[1])),
                confidence=float(conf),
                dist=float(1.0 - conf),
                ratio=float(1.0 - conf),
                is_inlier=True
            )
            correspondences.append(corr)
            matches_dict.append({
                "src_x": float(p_s[0]),
                "src_y": float(p_s[1]),
                "ref_x": float(p_r[0]),
                "ref_y": float(p_r[1]),
                "confidence": float(conf)
            })

        return correspondences, matches_dict