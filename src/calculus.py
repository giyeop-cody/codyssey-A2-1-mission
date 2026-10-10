"""요구 2(미적분): 수치미분의 오차 구조와 '등고선 ⊥ 그라디언트' 검사의 유효성.

원문 조건(감사 반영 — 이전 구현은 x=1.0 / f=x²+4y² 였다):
  2-1 f(x)=x², **x=3**, 해석적 미분 6 과 비교 → 오차 ≤1e-4
  2-2 **f(x,y)=x²+y²** 의 등고선 + 2-3 그라디언트 화살표 + 2-4 수직임 확인

두 가지 함정은 그대로 표로 남긴다.
1. f(x)=x^2 에서 중심차분은 **모든 h에서** 절단오차가 0이다(f'''=0). 즉 미션의 "≤1e-4" 검사는 항등식이다.
2. 등고선 접선을 그라디언트에서 유도하면(-Gy,Gx) 직교는 **정의상** 성립한다(항등식). 그래서 접선을
   matplotlib 이 그린 **등고선 다각형 좌표**에서 뽑아 실제 각도 편차를 재고, 유도로 만든 경우와 나란히 싣는다.
"""
from __future__ import annotations

import matplotlib
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np

from _common import REP, FIG, save, write

# 도함수는 x 에 의존한다 — 이전 버전은 `lambda x: 2.0 + 0*x`(=상수 2) 였다.
# x=1 에서만 맞던 그 식을 x=3 으로 옮기면 해석값이 2 로 남아 오차가 4.0 이 된다(감사에서 스펙을 x=3 으로
# 옮기며 드러난 잠복 버그). 원문 2-1 의 해석값은 2x = 6 이다.
FUNCS = {
    "x^2": (lambda x: x * x, lambda x: 2.0 * x),
    "x^3": (lambda x: x ** 3, lambda x: 3.0 * x * x),
    "sin(x)": (np.sin, np.cos),
}
HS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
X_SPEC = 3.0          # 원문 2-1 이 지정한 지점
X_EXTRA = 1.0         # 이전 구현이 쓰던 지점 — 민감도 대조로만 남긴다

# 원문 2-2 가 지정한 등고선 함수, 그리고 민감도 대조용 타원
CONTOURS = {
    "원문: f = x^2 + y^2": (lambda x, y: x * x + y * y, (-2.0, 2.0, 220), (-2.0, 2.0, 220), [0.5, 1.5, 3.0]),
    "추가(타원): f = x^2 + 4y^2": (lambda x, y: x * x + 4.0 * y * y, (-2.0, 2.0, 220), (-1.6, 1.6, 180), [1.0, 2.0, 3.5]),
}
SPEC_KEY = "원문: f = x^2 + y^2"


def central_diff(f, x: float, h: float) -> float:
    """중심차분 f'(x) ≈ (f(x+h) − f(x−h)) / 2h.

    테일러 전개 f(x±h) = f ± h·f′ + h²/2·f″ ± h³/6·f‴ + ... 를 빼면 짝수차 항이 지워져
    절단오차 = h²/6 · f‴(ξ) (O(h²)). 반올림오차 ≈ ε·|f|/h 이므로 총오차는 h ≈ (3ε|f|/|f‴|)^(1/3) 근처에서 최소.
    f=x² 이면 f‴=0 → 절단오차가 0 이라 원문 2-1 의 ≤1e-4 는 어떤 h 에서도 통과한다(검사력 없음).
    """
    return float((f(x + h) - f(x - h)) / (2.0 * h))


def error_table(x: float) -> list[dict]:
    rows = []
    for name, (f, df) in FUNCS.items():
        errs = []
        for h in HS:
            e = abs(central_diff(f, x, h) - float(df(x)))
            errs.append((h, e))
        best_h, best_e = min(errs, key=lambda t: t[1])
        rows.append({"함수": name, "x": x, "해석값": round(float(df(x)), 6),
                     "h_최적": f"{best_h:.0e}", "오차_최적": f"{best_e:.3e}",
                     "h=1e-1": f"{errs[0][1]:.3e}", "h=1e-8": f"{errs[-1][1]:.3e}",
                     "오차_최대(전h)": f"{max(e for _, e in errs):.3e}",
                     "1e-4_전구간통과": all(e <= 1e-4 for _, e in errs),
                     "gate_원문2-1(≤1e-4)": bool(all(e <= 1e-4 for _, e in errs)) if name == "x^2" else None,
                     "판독": "절단오차=0 → 검사 무의미" if name == "x^2" else "h^2 절단 + 1/h 반올림이 함께 보인다"})
    return rows


def numerical_gradient(f, p, h: float = 1e-5) -> np.ndarray:
    """다변수 수치 그라디언트(중심차분): ∂f/∂x_i ≈ (f(p + h e_i) − f(p − h e_i)) / 2h.  p: (n,) → (n,)."""
    p = np.asarray(p, dtype=float)
    g = np.zeros_like(p)
    for i in range(p.size):
        e = np.zeros_like(p); e[i] = h
        g[i] = (f(*(p + e)) - f(*(p - e))) / (2.0 * h)
    return g


def gradient_visualization(f, levels, xr=(-2.0, 2.0, 220), yr=(-2.0, 2.0, 220), spacing: float = 0.42,
                           ax=None, h: float = 1e-5):
    """원문 제출물 2 의 'Gradient 시각화 함수': 등고선을 그리고 **등고선 위의 점**에서 ∇f 화살표를 그린다.

    - 화살표 위치: matplotlib 이 그린 각 등고선 곡선 위에서 호 길이 spacing 간격의 점(원문 2-3 "등고선 위 여러 점").
      (등고선은 그림 영역 안에 닫히도록 levels 를 고른다 — 영역 밖으로 잘린 조각에 화살표가 몰리는 것을 피한다)
    - 화살표 방향: numerical_gradient(중심차분) — 해석식이 아니라 이 모듈의 수치미분으로 구한다.
    - quiver(angles='xy', scale_units='xy') + set_aspect('equal') → 화면상의 각도가 데이터 각도와 같다
      (aspect 가 같지 않으면 직교가 그림에서 비뚤어져 보인다).
    반환: (ax, 화살표 위치 (m,2), 그라디언트 (m,2), 등고선 객체)

    2026-10-11 정정: 이전 같은 이름의 함수는 1변수 x² 의 미분값 하나를 dict 로 돌려줄 뿐 **아무것도 그리지 않았고**,
    화살표는 별도 코드에서 np.gradient(격자 차분)로 격자점마다 찍었다(등고선 위가 아님).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.6))
    gx = np.linspace(*xr); gy = np.linspace(*yr)
    X, Y = np.meshgrid(gx, gy)
    cs = ax.contour(X, Y, f(X, Y), levels=levels, cmap="viridis")
    pts = []
    for segs in cs.allsegs:
        for path in segs:
            if len(path) < 12:
                continue
            d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
            for t in np.arange(0.0, d[-1] - 0.5 * spacing, spacing):
                pts.append([np.interp(t, d, path[:, 0]), np.interp(t, d, path[:, 1])])
    P = np.array(pts)
    G = np.array([numerical_gradient(f, p, h) for p in P])
    span = max(xr[1] - xr[0], yr[1] - yr[0])
    Gs = G / np.linalg.norm(G, axis=1, keepdims=True).max() * span * 0.12       # 길이는 |∇f| 에 비례(최댓값 정규화)
    ax.quiver(P[:, 0], P[:, 1], Gs[:, 0], Gs[:, 1], angles="xy", scale_units="xy", scale=1.0,
              color="#c0392b", width=0.006, zorder=3)
    ax.plot(P[:, 0], P[:, 1], "k.", ms=3, zorder=4)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    return ax, P, G, cs


def _perp_one(f, xr, yr, levels):
    ax_draw = plt.gca()
    gx = np.linspace(*xr); gy = np.linspace(*yr)
    X, Y = np.meshgrid(gx, gy)
    Z = f(X, Y)
    _tmpfig, _tmpax = plt.subplots()
    cs = _tmpax.contour(X, Y, Z, levels=levels)       # 측정용(보이지 않는 축) — 표시는 gradient_visualization
    plt.close(_tmpfig)
    devs_real, devs_taut = [], []
    singular = 0
    # 2026-10-11 정정: 이전에는 cs.allsegs[0](첫 등고선 1개)만 쟀다 → 그린 등고선 전부를 잰다
    for path in (pp for segs in cs.allsegs for pp in segs):
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
        nrm = np.linalg.norm(G, axis=1, keepdims=True)
        ok = (nrm[:, 0] > 1e-12)                       # 원점에서 ∇f=0 → 방향 미정의(제외하고 개수를 남긴다)
        singular += int((~ok).sum())
        G, Gn, tan, mid = G[ok], (G / nrm)[ok], tan[ok], mid[ok]
        if not len(Gn):
            continue
        cos = np.abs(np.einsum("ij,ij->i", Gn, tan))                # 직교하면 0
        devs_real.append(np.degrees(np.arccos(np.clip(cos, 0, 1))) - 90.0)   # 0에 가까울수록 직교
        # 항등식 버전: 접선을 (-Gy, Gx) 로 만들면 각도는 90° 가 정확히 나온다
        taut = np.stack([-G[:, 1], G[:, 0]], axis=1)
        taut = taut / np.linalg.norm(taut, axis=1, keepdims=True)
        devs_taut.append(np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", Gn, taut)), 0, 1))) - 90.0)
    # 요구 2-3: 등고선 **위의** 여러 점에서 Gradient 화살표 (gradient_visualization 이 그린다)
    _, P_arrow, G_arrow, cs_drawn = gradient_visualization(f, levels, xr, yr, ax=ax_draw)
    real = np.concatenate(devs_real) if devs_real else np.zeros(1)
    taut = np.concatenate(devs_taut) if devs_taut else np.zeros(1)
    return {"표본수": int(len(real)),
            "실측_최대편차_deg": f"{float(np.max(np.abs(real))):.4f}",
            "실측_중위편차_deg": f"{float(np.median(np.abs(real))):.4f}",
            "유도_최대편차_deg": f"{float(np.max(np.abs(taut))):.3e}",
            "특이점(∇f=0)_제외개수": singular,
            "화살표_개수(등고선 위)": int(len(P_arrow)),
            "화살표_위치_f값_범위": [round(float(min(f(*p) for p in P_arrow)), 4), round(float(max(f(*p) for p in P_arrow)), 4)]}, cs_drawn


def perpendicularity():
    """실측(그려진 등고선) vs 항등식(유도 접선) 각도 편차 — 원문 함수와 타원 대조를 하나의 Figure(서브플롯 2)로."""
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 4.4))
    res = {}
    for ax, (label, (f, xr, yr, levels)) in zip(axs, CONTOURS.items()):
        plt.sca(ax)
        stat, cs = _perp_one(f, xr, yr, levels)
        res[label] = stat
        ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"{label}\n실측 최대 {stat['실측_최대편차_deg']}° · 중위 {stat['실측_중위편차_deg']}° "
                     f"({stat['표본수']}점)", fontsize=8)
    axs[0].legend(handles=[cs.legend_elements()[0][0],
                           matplotlib.lines.Line2D([], [], color="#c0392b", lw=2, label="∇f 화살표")],
                  labels=["등고선", "∇f 화살표"], fontsize=7, loc="upper right")
    fig.suptitle("요구 2-2/2-3/2-4: 등고선과 그라디언트의 직교 (하나의 Figure, 서브플롯 2)", fontsize=9)
    save(fig, "03_gradient_contours.png")
    res["해석"] = ("유도 접선은 편차 0(항등식, 검사로서 가치 없음). 그려진 등고선에서 재면 격자 이산화·보간으로 "
                   "0.1~수 도 수준의 편차가 남 → '직교한다'고 주장하려면 이 편차와 격자 해상도를 함께 적어야 한다. "
                   "원점에서는 ∇f=0 이라 방향이 정의되지 않아 표본에서 제외했고 그 개수를 적는다")
    return res


def main() -> dict:
    rows = error_table(X_SPEC)
    rows_extra = error_table(X_EXTRA)
    perp = perpendicularity()
    # 그라디언트 시각화 함수의 수치 요약: 원문 함수 x²+y² 의 몇 점에서 수치 ∇f 와 해석 ∇f=(2x,2y) 비교
    f_spec = CONTOURS[SPEC_KEY][0]
    probe = np.array([[1.0, 1.0], [-1.2, 0.5], [0.3, -1.9], [3.0, 0.0]])
    gnum = np.array([numerical_gradient(f_spec, q) for q in probe])
    gv = {"함수": "f(x,y)=x²+y²", "점": probe.tolist(), "수치_∇f": np.round(gnum, 8).tolist(),
          "해석_∇f=(2x,2y)": (2 * probe).tolist(),
          "최대_절대오차": f"{float(np.max(np.abs(gnum - 2 * probe))):.3e}",
          "그림": "figures/03_gradient_contours.png (화살표 = numerical_gradient, 위치 = 등고선 위 등간격 점)"}
    x2 = [r for r in rows if r["함수"] == "x^2"][0]
    spec, extra = perp[SPEC_KEY], perp["추가(타원): f = x^2 + 4y^2"]
    out = {"표": rows,
           "원문2-1_검증": {"지점": f"x={X_SPEC:g}(원문 지정)", "해석값": 6.0,
                            "오차_최대(전h)": x2["오차_최대(전h)"], "gate_≤1e-4": x2["1e-4_전구간통과"],
                            "판독": x2["판독"]},
           "민감도_지점(x=1.0)": [r for r in rows_extra if r["함수"] == "x^2"],
           "그라디언트_시각화": gv,
           "직교성": perp,
           "판정": (f"원문 조건(x=3, 해석값 6)에서 x^2 중심차분 오차 최대 {x2['오차_최대(전h)']} → ≤1e-4 통과. "
                    f"그러나 f'''=0 이라 **어떤 h 에서도** 통과한다(전 h 통과={x2['1e-4_전구간통과']}) "
                    "→ 미션의 오차 조건은 이 함수에서 **검증이 아니다**. 구분력은 x^3·sin 에서 나온다. "
                    f"등고선 직교는 원문 함수 x²+y² 에서 실측 최대 {spec['실측_최대편차_deg']}°"
                    f"(타원 x²+4y² 는 {extra['실측_최대편차_deg']}° — 격자·곡률에 따라 편차가 커진다)"),
           "조건": (f"수치미분: f=f(x) 는 **x={X_SPEC:g}**(원문 지정, 해석값 6) · 중심차분 · float64 · "
                    f"h 1e-1..1e-8 (8종). 대조로 x={X_EXTRA:g} 도 싣는다. "
                    "등고선: matplotlib contour(levels=[0.5,1.5,3] for x²+y², [1,2,3.5] for 타원 — 영역 안에서 닫히는 값, 2026-10-11 변경) · 원문 함수 x²+y² 는 220×220 격자(x,y ∈ [-2,2]), "
                    "타원 x²+4y² 는 220×180(x ∈ [-2,2], y ∈ [-1.6,1.6])")}
    return write("calculus.json", out)


if __name__ == "__main__":
    d = main()
    for r in d["표"]:
        print(r)
    print(d["원문2-1_검증"])
    print(d["직교성"][SPEC_KEY])
    print(d["판정"])
