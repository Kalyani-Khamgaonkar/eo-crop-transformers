"""2 x 6 results table for the 6 Nov screening (val macro mIoU, mean +/- std over seeds).

    python scripts/analyse_screen.py

Sources: pe_screen_6nov (new cells), nine_rest (C2 = 2D + conv, C5 = tubelet + conv),
nine_c1_c4 (old C1 / C4, date table with the old N(0,1) start, shown for comparison),
scale_20pct (optional). Baseline (2D + date sinusoid) is the Stage 4 result.
"""
import re
import statistics as st

import yaml

EP = re.compile(r"\[ep\s+(\d+)\] loss ([\d.]+)/([\d.]+)\s+mIoU ([\d.]+)/([\d.]+)\s+OA ([\d.]+)/([\d.]+)\s+(\d+)s")
BASELINE = (0.454, 0.015, 3)


def collect(exp, key=lambda rid: rid.rsplit("_s", 1)[0]):
    res = {}
    try:
        runs = yaml.safe_load(open(f"experiments/{exp}.yaml"))["runs"]
    except FileNotFoundError:
        return res
    for i, r in enumerate(runs):
        try:
            t = open(f"logs/run_{exp}_{i}.out").read()
        except FileNotFoundError:
            continue
        m = re.search(r"best val mIoU ([\d.]+) at epoch (\d+)", t)
        if not m:
            continue
        ep = {int(e[0]): [float(x) for x in e[1:]] for e in EP.findall(t)}
        b = int(m.group(2))
        res.setdefault(key(r["id"]), []).append(dict(
            val=float(m.group(1)),
            gap=ep[b][2] - ep[b][3] if b in ep else float("nan"),
            oa=ep[b][5] if b in ep else float("nan"),
            sec=st.median(v[6] for v in ep.values()) if ep else float("nan")))
    return res


def fmt(v):
    if not v:
        return "-"
    x = [d["val"] for d in v]
    return f"{st.mean(x):.3f}" + (f" +/- {st.stdev(x):.3f}" if len(x) > 1 else "") + f" (n={len(x)})"


def main():
    new = collect("pe_screen_6nov")
    old9 = collect("nine_rest", key=lambda rid: rid[:2])
    old14 = collect("nine_c1_c4", key=lambda rid: rid[:2])
    cells = {
        ("2D", "E0"): new.get("E0_2d"), ("Tubelet", "E0"): new.get("E0_tube"),
        ("2D", "E1"): None, ("Tubelet", "E1"): new.get("E1_tube"),
        ("2D", "E2"): new.get("C1b_2d_E2"), ("Tubelet", "E2"): new.get("C4b_tube_E2"),
        ("2D", "E3"): old9.get("C2"), ("Tubelet", "E3"): old9.get("C5"),
        ("2D", "E4"): new.get("E4_2d"), ("Tubelet", "E4"): new.get("E4_tube"),
        ("2D", "E5"): new.get("E5_2d"), ("Tubelet", "E5"): new.get("E5_tube"),
    }
    cols = [("E0", "None"), ("E1", "Date sinusoid"), ("E2", "Date table"), ("E3", "Conv PE"),
            ("E4", "Frame idx concat"), ("E5", "3D rel. bias")]
    print("Validation macro mIoU, 5% subset, fold I (mean +/- std over seeds)\n")
    print(f"{'':9}" + "".join(f"{c + ' ' + n:26}" for c, n in cols))
    for row in ("2D", "Tubelet"):
        line = f"{row:9}"
        for c, _ in cols:
            if (row, c) == ("2D", "E1"):
                line += f"{f'{BASELINE[0]:.3f} +/- {BASELINE[1]:.3f} (n=3)':26}"
            else:
                line += f"{fmt(cells[(row, c)]):26}"
        print(line)
    print("\nDate table, old start (N(0,1)) vs new start (std 0.02):")
    print(f"  2D      old C1 {fmt(old14.get('C1')):24} new C1b {fmt(new.get('C1b_2d_E2'))}")
    print(f"  Tubelet old C4 {fmt(old14.get('C4')):24} new C4b {fmt(new.get('C4b_tube_E2'))}")
    print("\nid                 n  val mIoU          gap    OA     s/epoch")
    for name, v in sorted({**new, **{k: old9[k] for k in ("C2", "C5") if k in old9}}.items()):
        x = [d["val"] for d in v]
        print(f"{name:18} {len(x)}  {st.mean(x):.3f} +/- {st.stdev(x) if len(x) > 1 else 0:.3f}  "
              f"{st.mean(d['gap'] for d in v):.3f}  {st.mean(d['oa'] for d in v):.3f}  {st.mean(d['sec'] for d in v):.0f}")
    sc = collect("scale_20pct")
    if sc:
        print("\n20% subset (seed 42):")
        for k, v in sc.items():
            print(f"  {k:20} {fmt(v)}")


if __name__ == "__main__":
    main()
