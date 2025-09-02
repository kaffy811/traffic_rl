# scripts/plot_ci_and_overall.py
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT = Path("reports/SUMMARY_JSON")
CI_CSV = ROOT / "per_port_wilson_ci.csv"
assert CI_CSV.exists(), f"Missing {CI_CSV} (先运行 stats_sigcheck.py 生成)"

# --------- helpers ---------
def wilson_ci(p_hat: float, n: int, z: float = 1.96):
    if n <= 0:
        return 0.0, 0.0
    denom = 1 + z*z/n
    centre = p_hat + z*z/(2*n)
    margin = z * ((p_hat*(1-p_hat)/n + z*z/(4*n*n))**0.5)
    lo = max(0.0, (centre - margin)/denom)
    hi = min(1.0, (centre + margin)/denom)
    return lo, hi

def pick(d: dict, names):
    for name in names:
        if name in d and d[name] not in (None, "", "nan"):
            return d[name]
    return None

def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def to_int(x, default=None):
    try:
        return int(float(x))
    except Exception:
        return default

# --------- load CSV (robust to header variants) ---------
rows = []
with CI_CSV.open("r", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for raw in rdr:
        r = {k.strip().lower(): (raw[k].strip() if isinstance(raw[k], str) else raw[k])
             for k in raw.keys()}

        port = pick(r, ["port"])
        tag  = pick(r, ["tag", "strategy"])

        p_hat = to_float(pick(r, ["p_hat", "phat", "p"]))
        n     = to_int(pick(r, ["n", "num_samples", "samples"]))

        # 多种命名的低/高界
        lo = to_float(pick(r, ["ci_low", "ci_lo", "low", "lower", "lb", "wilson_lb"]))
        hi = to_float(pick(r, ["ci_high", "ci_hi", "high", "upper", "ub", "wilson_hi", "upper_bound"]))

        if p_hat is None or n is None:
            # 跳过无法解析的行
            continue

        if lo is None or hi is None:
            lo, hi = wilson_ci(p_hat, n)

        rows.append(dict(port=port, tag=tag, p_hat=p_hat, n=n, lo=lo, hi=hi))

# 固定顺序
port_order = ["baton_rouge", "new_orleans", "south_louisiana", "gulfport"]
tag_order  = ["Balanced200", "Unbalanced200", "FairUnbalanced200"]

# 组织成: data[port][tag] = (p_hat, lo, hi, n)
data = defaultdict(dict)
for r in rows:
    if r["port"] and r["tag"]:
        data[r["port"]][r["tag"]] = (r["p_hat"], r["lo"], r["hi"], r["n"])

# --------- Plot 1: per-port bars with Wilson CI ---------
fig1 = plt.figure(figsize=(10, 5.5))
xbase = list(range(len(port_order)))
width = 0.25

for i, tag in enumerate(tag_order):
    xs = [x + (i - 1)*width for x in xbase]
    ys, yerr_low, yerr_high = [], [], []
    for port in port_order:
        p_hat, lo, hi, _ = data.get(port, {}).get(tag, (0.0, 0.0, 0.0, 0))
        ys.append(p_hat)
        yerr_low.append(max(0.0, p_hat - lo))
        yerr_high.append(max(0.0, hi - p_hat))
    plt.bar(xs, ys, width=width, label=tag)
    plt.errorbar(xs, ys, yerr=[yerr_low, yerr_high], fmt="none", capsize=3)

plt.xticks(xbase, port_order, rotation=15)
plt.ylabel("Success rate")
plt.title("Per-port success rate with Wilson 95% CI")
plt.ylim(0.0, 1.0)
plt.legend()
out1 = ROOT / "fig_wilson_errorbars.png"
plt.tight_layout()
plt.savefig(out1, dpi=150)
plt.close(fig1)

# --------- Plot 2/3: Overall Micro/Macro for Unbalanced vs Fair ---------
def micro(tag):
    num, den = 0.0, 0
    for p in port_order:
        if tag in data.get(p, {}):
            ph, _, _, n = data[p][tag]
            num += ph * n
            den += n
    return num/den if den > 0 else 0.0

def macro(tag):
    vals = []
    for p in port_order:
        if tag in data.get(p, {}):
            ph, _, _, _ = data[p][tag]
            vals.append(ph)
    return sum(vals)/len(vals) if vals else 0.0

tags2 = ["Unbalanced200", "FairUnbalanced200"]

fig2 = plt.figure(figsize=(5, 4))
plt.bar(tags2, [micro(t) for t in tags2])
plt.ylim(0.0, 1.0)
plt.ylabel("Success rate")
plt.title("Overall (Micro-average)")
out2 = ROOT / "fig_overall_micro.png"
plt.tight_layout()
plt.savefig(out2, dpi=150)
plt.close(fig2)

fig3 = plt.figure(figsize=(5, 4))
plt.bar(tags2, [macro(t) for t in tags2])
plt.ylim(0.0, 1.0)
plt.ylabel("Success rate")
plt.title("Overall (Macro-average)")
out3 = ROOT / "fig_overall_macro.png"
plt.tight_layout()
plt.savefig(out3, dpi=150)
plt.close(fig3)

print("Saved:\n -", out1, "\n -", out2, "\n -", out3)
