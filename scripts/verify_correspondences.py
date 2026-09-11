"""
Manual Verification Support Tool (SIH26166 Section 7).
Enables interactive inspection and validation of candidate correspondences.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np

def inspect_pair(src_path: Path, ref_path: Path, points_path: Path = None):
    print(f"[*] Inspecting: {src_path.name} <-> {ref_path.name}")
    s_img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    r_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)

    if s_img is None or r_img is None:
        print("[!] Error loading images.")
        return

    h = max(s_img.shape[0], r_img.shape[0])
    w_s, w_r = s_img.shape[1], r_img.shape[1]
    canvas = np.zeros((h, w_s + w_r, 3), dtype=np.uint8)
    canvas[:s_img.shape[0], :w_s] = cv2.cvtColor(s_img, cv2.COLOR_GRAY2BGR)
    canvas[:r_img.shape[0], w_s:] = cv2.cvtColor(r_img, cv2.COLOR_GRAY2BGR)

    out_path = Path("results") / "inspection_preview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"[+] Saved inspection visualization to {out_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect and verify correspondence pairs")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--reference", type=str, required=True)
    args = parser.parse_args()
    inspect_pair(Path(args.source), Path(args.reference))
