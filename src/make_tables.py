"""Collect results/*.json into a markdown table and a LaTeX table."""
import argparse
import glob
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="random", choices=["random", "dedup"])
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    res_dir = os.path.join(args.results_dir, args.split)
    rows = []
    for path in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        t = r["test"]
        rows.append((r["model"], r["val"]["f1_macro"], t["precision_macro"], t["recall_macro"], t["f1_macro"],
                     t["accuracy"], t["f1_negative"], t["f1_neutral"], t["f1_positive"]))
    rows.sort(key=lambda x: x[4])

    md = ["| Model | Val F1 | Test P | Test R | Test F1 | Test Acc | F1 neg | F1 neu | F1 pos |",
          "|---|---|---|---|---|---|---|---|---|"]
    tex = []
    for name, *vals in rows:
        nums = [f"{100 * v:.2f}" for v in vals]
        md.append(f"| {name} | " + " | ".join(nums) + " |")
        tex.append(name.replace("_", r"\_") + " & " + " & ".join(nums[1:5]) + r" \\")
    text = "\n".join(md)
    print(text)
    with open(os.path.join(res_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(res_dir, "summary.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
