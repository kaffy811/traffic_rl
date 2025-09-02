#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 nightly/forced_*.json 聚合为每港口的加权成功率，并给出
- Wilson 95% CI
- Fair vs Unbalanced 的两比例 z 检验（双侧 p 值）
输出到控制台，并把 CSV 存到 reports/SUMMARY_JSON/
"""
import os, glob, json, math
from pathlib import Path
from collections import defaultdict

def wilson_ci(p, n, z=1.96):
    if n <= 0: return (0.0, 0.0)
    denom = 1 + z*z/n
    center = p + z*z/(2*n)
    adj = z * math.sqrt((p*(1-p) + z*z/(4*n))/n)
    lo = max(0.0, (center - adj)/denom)
    hi = min(1.0, (center + adj)/denom)
    return lo, hi

def two_prop_ztest(p1, n1, p2, n2):
    if min(n1, n2) <= 0: return float('nan')
    x1, x2 = p1*n1, p2*n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    if se == 0: return float('inf'), 0.0
    z = (p1 - p2)/se
    # 近似双侧 p 值
    # 正态分布尾部：p≈2*(1-Φ(|z|)); 这里用简化近似
    # 误差对我们展示用足够
    from math import erf, sqrt
    cdf = 0.5*(1+erf(z/math.sqrt(2)))
    p = 2*min(cdf, 1-cdf)
    return z, p

def load_tag(tag):
    rows = defaultdict(lambda: {"succ":0.0, "n":0})
    fs = glob.glob(f"reports/FLW_{tag}/nightly/forced_*.json")
    for f in fs:
        o = json.load(open(f))
        port = o["port"]
        n = int(o.get("num_samples") or o.get("samples") or 0)
        sr = o.get("success_rate")
        if n and (sr is not None):
            rows[port]["succ"] += float(sr)*n
            rows[port]["n"]    += n
    agg = {}
    for port, v in rows.items():
        n = int(v["n"])
        p = (v["succ"]/n) if n>0 else 0.0
        lo, hi = wilson_ci(p, n)
        agg[port] = {"p": p, "n": n, "lo": lo, "hi": hi}
    return agg

def main():
    out_dir = Path("reports/SUMMARY_JSON"); out_dir.mkdir(parents=True, exist_ok=True)
    BAL  = os.getenv("BAL")
    UNB  = os.getenv("UNB")
    FAIR = os.getenv("FAIR")
    if not all([BAL,UNB,FAIR]):
        raise SystemExit("请先 export BAL/UNB/FAIR 三个 tag 环境变量")

    bal  = load_tag(BAL)
    unb  = load_tag(UNB)
    fair = load_tag(FAIR)
    ports = sorted(set(bal)|set(unb)|set(fair))

    # 屏幕表格
    print("\n=== Per-port Wilson 95% CI ===")
    print("port, tag, p_hat, n, CI_low, CI_high")
    for tag_name, agg in [("Balanced200", bal), ("Unbalanced200", unb), ("FairUnbalanced200", fair)]:
        for port in ports:
            if port not in agg: continue
            a = agg[port]
            print(f"{port}, {tag_name}, {a['p']:.4f}, {a['n']}, {a['lo']:.4f}, {a['hi']:.4f}")

    # 显著性：Fair vs Unb
    lines = ["port, p_fair, n_fair, p_unb, n_unb, delta, z, pvalue"]
    for port in ports:
        if port in fair and port in unb:
            pf, nf = fair[port]["p"], fair[port]["n"]
            pu, nu = unb[port]["p"],  unb[port]["n"]
            z, p = two_prop_ztest(pf, nf, pu, nu)
            lines.append(f"{port},{pf:.6f},{nf},{pu:.6f},{nu},{(pf-pu):.6f},{z:.3f},{p:.4f}")
    (out_dir / "sig_fair_vs_unb.csv").write_text("\n".join(lines), encoding="utf-8")
    print("\n=== Saved:", out_dir / "sig_fair_vs_unb.csv")

    # 同时写一份 per-port CI CSV，便于论文附录
    ci_lines = ["port,tag,p_hat,n,lo,hi"]
    for tag_name, agg in [("Balanced200", bal), ("Unbalanced200", unb), ("FairUnbalanced200", fair)]:
        for port in ports:
            if port not in agg: continue
            a = agg[port]
            ci_lines.append(f"{port},{tag_name},{a['p']:.6f},{a['n']},{a['lo']:.6f},{a['hi']:.6f}")
    (out_dir / "per_port_wilson_ci.csv").write_text("\n".join(ci_lines), encoding="utf-8")
    print("=== Saved:", out_dir / "per_port_wilson_ci.csv")

if __name__ == "__main__":
    main()
