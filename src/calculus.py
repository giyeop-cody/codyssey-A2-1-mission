"""요구 2(미적분): 수치미분의 오차 구조와 '등고선 ⊥ 그라디언트' 검사의 유효성.

두 가지 함정을 표로 남긴다.
1. f(x)=x^2 에서 중심차분은 **모든 h에서** 절단오차가 0이다(f'''=0). 즉 미션의 "≤1e-4" 검사는 항등식이다.
2. 등고선 접선을 그라디언트에서 유도하면(-Gy,Gx) 직교는 **정의상** 성립한다(항등식). 그래서 접선을
   matplotlib 이 그린 **등고선 다각형 좌표**에서 뽑아 실제 각도 편차를 재고, 유도로 만든 경우와 나란히 싣는다.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import REP, FIG, save, write

FUNCS = {
    "x^2": (lambda x: x * x, lambda x: 2.0 + 0 * x),
    "x^3": (lambda x: x ** 3, lambda x: 3.0 * x * x),
    "sin(x)": (np.sin, np.cos),
}
HS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


def central_diff(f, x: float, h: float) -> float:
    return float((f(x + h) - f(x - h)) / (2.0 * h))


def error_table(x: float = 1.0) -> list[dict]:
    rows = []
    for name, (f, df) in FUNCS.items():
        errs = []
        for h in HS:
            e = abs(central_diff(f, x, h) - float(df(x)))
            errs.append((h, e))
        best_h, best_e = min(errs, key=lambda t: t[1])
        small_h = max(errs[-3:], key=lambda t: t[1])       # h 가 극단적으로 작으면 반올림 오차가 지배
        rows.append({"함수": name, "h_최적": f"{best_h:.0e}", "오차_최적": f"{best_e:.3e}",
                     "h=1e-1": f"{errs[0][1]:.3e}", "h=1e-8": f"{errs[-1][1]:.3e}",
                     "1e-4_전구간통과": all(e <= 1e-4 for _, e in errs),
                     "판독": "절단오차=0 → 검사 무의미" if name == "x^2" else "h^2 절단 + 1/h 반올림이 함께 보인다"})
        del small_h
    return rows


def perpendicularity():
    """실측(그려진 등고선) vs 항등식(유도 접선) 각도 편차."""
    f = lambda x, y: x * x + 4.0 * y * y
    gx = np.linspace(-2, 2, 220)
    gy = np.linspace(-1.6, 1.6, 180)
    X, Y = np.meshgrid(gx, gy)
    Z = f(X, Y)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    cs = ax.contour(X, Y, Z, levels=[2.0, 6.0, 10.0])
    devs_real, devs_taut = [], []
    for path in cs.allsegs[0]:
        if len(path) < 12:
            continue
        step = path[1:] - path[:-1]
        L = np.hypot(step[:, 0], step[:, 1])
        keep = L > 1e-9
        mid = (path[:-1][keep] + path[1:][keep]) / 2.0
        tan = step[keep] / L[keep][:, None]
        Gx = np.array([central_diff(lambda t: f(t, m[1]), m[0], 1e-5) for m in mid])
        Gy = np.array([central_diff(lambda t: f(m[0], t), m[1], 1e-5) for m in mid])
        G = np.stack([Gx, Gy], axis=1)
        Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
        cos = np.abs(np.einsum("ij,ij->i", Gn, tan))                # 직교하면 0
        devs_real.append(np.degrees(np.arccos(np.clip(cos, 0, 1))) - 90.0)   # 0에 가까울수록 직교
        # 항등식 버전: 접선을 (-Gy, Gx) 로 만들면 각도는 90° 가 정확히 나온다
        taut = np.stack([-G[:, 1], G[:, 0]], axis=1)
        taut = taut / np.linalg.norm(taut, axis=1, keepdims=True)
        devs_taut.append(np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", Gn, taut)), 0, 1))) - 90.0)
    real = np.concatenate(devs_real)
    taut = np.concatenate(devs_taut)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    Gy_g, Gx_g = np.gradient(Z, gy[1] - gy[0], gx[1] - gx[0])
    ax.quiver(X[::24, ::24], Y[::24, ::24], Gx_g[::24, ::24], Gy_g[::24, ::24],
              color="#c0392b", scale=40, width=0.006)
    ax.set_title("등고선과 그라디언트(실측 접선 vs 유도 접선)")
    save(fig, "03_gradient_contours.png")
    return {"표본수": int(len(real)),
            "실측_최대편차_deg": f"{float(np.max(np.abs(real))):.4f}",
            "실측_중위편차_deg": f"{float(np.median(np.abs(real))):.4f}",
            "유도_최대편차_deg": f"{float(np.max(np.abs(taut))):.3e}",
            "해석": "유도 접선은 편차 0(항등식, 검사로서 가치 없음). 그려진 등고선에서 재면 격자 이산화·보간으로 "
                    "0.1~수 도 수준의 편차가 남 → '직교한다'고 주장하려면 이 편차와 격자 해상도를 함께 적어야 한다"}


def main() -> dict:
    rows = error_table()
    perp = perpendicularity()
    x2_all = [r for r in rows if r["함수"] == "x^2"][0]
    out = {"표": rows, "직교성": perp,
           "판정": f"x^2 은 전 h 구간에서 ≤1e-4 통과({x2_all['1e-4_전구간통과']}) → 미션의 오차 조건은 이 함수에서 **검증이 아니다**",
           "조건": "f=f(x) 는 x=1.0 지점, 중심차분, float64; 등고선은 matplotlib contour(levels=[2,6,10]) · 220×180 격자"}
    return write("calculus.json", out)


if __name__ == "__main__":
    d = main()
    for r in d["표"]:
        print(r)
    print(d["직교성"])
    print(d["판정"])
