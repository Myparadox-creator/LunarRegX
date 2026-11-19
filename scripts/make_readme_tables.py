"""
Generate Markdown tables for README.md and Technical Report from benchmark and ablation CSVs.
Ensures zero hand-written or invented metrics: all values come directly from results/.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def generate_benchmark_markdown_table(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)

    # Scenarios mapping
    scenarios = [
        ("Pair 1: Low Illumination Delta", "1. Baseline Control (Low $\\Delta\\text{Az}$)"),
        ("Pair 2: 180° Shadow Reversal", "2. Extreme 180° Shadow Flip"),
        ("Pair 3: Multi-Scale (OHRC vs LROC)", "3. Multi-Scale (0.25m vs 0.50m)"),
        ("Pair 4: Viewpoint / Affine Shear", "4. Oblique Viewpoint Shear"),
        ("Pair 5: Polar Crater Deep Shadow", "5. Polar Shadow & Noise"),
    ]

    # Engines to include
    engines = [
        ("Proposed (GIS-Assisted Hybrid)", "Proposed Phase-Structural"),
        ("Baseline 5a (LoFTR Pretrained)", "LoFTR (pretrained)"),
        ("Baseline 4 (RIFT2)", "RIFT2"),
        ("Baseline 1 (SIFT + RANSAC)", "SIFT (Baseline 1)"),
        ("Baseline 2 (AKAZE)", "AKAZE (Baseline 2)"),
        ("Baseline 3 (Learned CNN)", "Learned LunarNet"),
    ]

    headers = ["Scenario"] + [name for _, name in engines]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| :--- | " + " | ".join([":---:"] * len(engines)) + " |"

    rows = []
    engine_stats = {m_key: {"inlier_ratios": [], "rmses": []} for m_key, _ in engines}

    for scen_raw, scen_label in scenarios:
        row_cells = [f"**{scen_label}**"]
        scen_df = df[df["scenario"] == scen_raw]

        # Find best method in this scenario: prioritize SUCCESS over WARNING
        best_rmse = 999.0
        best_method = None
        # First check SUCCESS
        for m_key, _ in engines:
            sub = scen_df[scen_df["method"] == m_key]
            if not sub.empty:
                r = sub.iloc[0]
                if r["status"] == "SUCCESS" and r["inlier_count"] >= 10:
                    if float(r["rmse_pixels"]) < best_rmse:
                        best_rmse = float(r["rmse_pixels"])
                        best_method = m_key
        # If no SUCCESS, check WARNING
        if best_method is None:
            for m_key, _ in engines:
                sub = scen_df[scen_df["method"] == m_key]
                if not sub.empty:
                    r = sub.iloc[0]
                    if r["status"] == "WARNING" and r["inlier_count"] >= 10:
                        if float(r["rmse_pixels"]) < best_rmse:
                            best_rmse = float(r["rmse_pixels"])
                            best_method = m_key

        for m_key, _ in engines:
            sub = scen_df[scen_df["method"] == m_key]
            if sub.empty:
                row_cells.append("N/A")
                continue
            r = sub.iloc[0]
            status = str(r["status"]).upper()
            inl = int(r["inlier_count"])
            ratio = float(r["inlier_ratio"])
            rmse = float(r["rmse_pixels"])

            engine_stats[m_key]["inlier_ratios"].append(ratio)
            if status in ["SUCCESS", "WARNING"] and inl >= 10:
                engine_stats[m_key]["rmses"].append(rmse)

            if status == "FAILURE" or inl < 10:
                cell = f"❌ Failed ({inl} Inl, {ratio:.1%})"
            else:
                formatted = f"{rmse:.2f} px ({ratio:.1%} Inl)"
                if m_key == best_method:
                    cell = f"**{formatted}**"
                else:
                    cell = formatted
            row_cells.append(cell)
        rows.append("| " + " | ".join(row_cells) + " |")

    # Summary row: Average Inlier Ratio
    avg_inl_cells = ["**Average Inlier Ratio**"]
    for m_key, _ in engines:
        ratios = engine_stats[m_key]["inlier_ratios"]
        avg_r = np.mean(ratios) if ratios else 0.0
        avg_inl_cells.append(f"{avg_r:.1%}")
    rows.append("| " + " | ".join(avg_inl_cells) + " |")

    # Summary row: Average Successful RMSE
    avg_rmse_cells = ["**Average Inlier RMSE**"]
    for m_key, _ in engines:
        rmses = engine_stats[m_key]["rmses"]
        if rmses:
            avg_rmse_cells.append(f"{np.mean(rmses):.2f} px")
        else:
            avg_rmse_cells.append("N/A (Failed)")
    rows.append("| " + " | ".join(avg_rmse_cells) + " |")

    return "\n".join([header_line, sep_line] + rows)

def generate_ablation_markdown_table(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    header = "| Ablation Stage | Inliers | Inlier Ratio | RMSE (px) | Grid Coverage | Status |"
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: |"
    rows = []
    for _, r in df.iterrows():
        stage = r["Stage"]
        inl = int(r["Inliers"])
        ratio = float(r["Inlier_Ratio"])
        rmse = float(r["RMSE_px"])
        cov = float(r["Coverage"])
        st = str(r["Status"])
        rmse_str = f"{rmse:.3f} px" if rmse < 900 and inl >= 10 else "N/A"
        rows.append(f"| **{stage}** | {inl} | {ratio:.1%} | {rmse_str} | {cov:.1%} | {st} |")
    return "\n".join([header, sep] + rows)

def main():
    root = Path(__file__).resolve().parent.parent
    bench_csv = root / "results" / "benchmark_results.csv"
    ablation_csv = root / "results" / "ablation_results.csv"

    print("=== BENCHMARK TABLE ===")
    if bench_csv.exists():
        bench_md = generate_benchmark_markdown_table(bench_csv)
        print(bench_md)
    else:
        print(f"Error: {bench_csv} not found")

    print("\n=== ABLATION TABLE ===")
    if ablation_csv.exists():
        ablation_md = generate_ablation_markdown_table(ablation_csv)
        print(ablation_md)
    else:
        print(f"Error: {ablation_csv} not found")

if __name__ == "__main__":
    main()
