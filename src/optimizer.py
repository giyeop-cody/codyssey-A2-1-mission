"""요구 4(최적화): 경사하강 vs 모멘텀, 그리고 '수렴'을 주장할 때 붙어야 할 조건.

이차형식 f(p) = ½ pᵗA p (A 대각) 에서 GD는 **0 < lr < 2/λmax** 여야 발산하지 않는다.
원(A=diag(2,2), λmax=2)과 타원(A=diag(2,20), λmax=20)에서 상한이 1.0 vs 0.1 로 10배 차이 →
같은 lr=0.5 가 한쪽은 수렴, 한쪽은 폭발한다. 그래서 'lr 스윕'은 함수마다 따로 재야 의미가 있다.

감사(2026-09-27) 반영 4건:
1. 판정 라벨을 실제 거동과 맞춘다 — 이전엔 f<1e-8 이 아니면 전부 "미수렴(진동)"이라 붙였지만,
   원 lr=0.01·0.99 는 부호 반전 없는 **단조 수렴 중**(정체)이다. → 발산 / 진동(좌표 부호반전 관측) / 정체 / 수렴 4분할.
2. **원문 기준 열**을 추가한다: 원문 4-2 는 "lr=0.1, 100회 후 반경 0.1 이내"를 요구한다. 자체 기준(f<1e-8)은
   원문보다 훨씬 엄격하므로 두 기준을 나란히 싣는다.
3. 모멘텀·Adam 의 이론 상한을 정정한다(2차 감사에서 재 정정). 이 구현의 모멘텀은 **heavy-ball 변형**(v←βv+g, p←p−lr·v)이고,
   **발산 경계는 안정 한계 2(1+β)/λmax(H)**(원 1.9·타원 0.19 — 상태행렬 스펙트럼반경<1 ⇔ lr·λ<2(1+β), 실측 발산 경계와 일치)다.
   실효 학습률 lr/(1−β)=10·lr 에서 나오는 2(1−β)/λmax(H)(원 0.1·타원 0.01)는 "GD 동급 수렴속도" 기준이지 발산 경계가 아니고,
   초판의 2/λmax(H)(원 1.0·타원 0.1)는 둘 다 아니다.
4. 원문 4-4 "lr≥0.5 에서 발산"은 **원(x²+y²)에서는 성립하지 않는다**(이론 상한 1.0) → 대조 표를 싣는다.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import save, write

# A 는 f = ½ pᵗA p 의 행렬이다 → **헤세 H = A**. (이전 주석은 "A=diag(2,2) → λmax=2 → 상한 1.0"이라 썼는데,
# 코드에서 λmax 로 쓰는 값은 np.linalg.eigvalsh(A)[-1]=2·hess 이다. 숫자는 우연히 맞았지만 공식 표기가
# 인수(인자의 정의)와 어긋났다 → 아래 표기는 헤세 기준으로 통일한다.)
FUNCS = {"원: x^2 + y^2": np.diag([2.0, 2.0]), "타원: x^2 + 10y^2": np.diag([2.0, 20.0])}
# 헤세 최대고윳값: 원 H=2I → 2, 타원 H=diag(2,20) → 20
HESS = {"원: x^2 + y^2": 2.0, "타원: x^2 + 10y^2": 20.0}
LRS = [0.01, 0.05, 0.09, 0.095, 0.1, 0.15, 0.2, 0.4, 0.5, 0.9, 0.99, 1.0, 1.5]
P0 = np.array([-3.0, 2.0])
ITERS = 100
BETA = 0.9
F_CONV = 1e-8            # 자체 기준(수렴 선언)
R_SPEC = 0.1             # 원문 4-2 기준(반경)


class VanillaGD:
    name = "vanilla"
    upper = "2/λmax(H)"

    def __init__(self, lr: float):
        self.lr = lr

    def step(self, p, g):
        return p - self.lr * g


class Momentum:
    """heavy-ball 변형: v ← βv + g, p ← p − lr·v  (Nesterov 아님). 정상상태 실효 학습률 ≈ lr/(1−β).

    두 상한을 구분한다(2차 감사 자기정정):
    - 발산 경계 = **안정 한계** lr·λmax < 2(1+β): 좌표별 상태행렬 [[1−lr·λ, −lr·β],[λ, β]] 의 스펙트럼반경<1 조건.
    - 실효 lr = lr/(1−β) 휴리스틱의 2(1−β)/λmax 는 "GD 동급 수렴속도" 상한이지 발산 경계가 아니다.
    """
    name = "momentum"
    upper = "2(1+β)/λmax(H) (heavy-ball 안정 한계)"

    def __init__(self, lr: float, beta: float = BETA):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, p, g):
        self.v = g.copy() if self.v is None else self.beta * self.v + g
        return p - self.lr * self.v


class Adam:
    """보너스(요구문에 없으나 맵의 보너스 1순위). lr 는 1/10 로 줄여 쓴다(스케일 차이 방지)."""
    name = "adam"
    upper = "해석 상한 없음(적응형 스케일) — 측정보고용"

    def __init__(self, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.c = None
        self.t = 0

    def step(self, p, g):
        self.t += 1
        self.m = g.copy() if self.m is None else self.b1 * self.m + (1 - self.b1) * g
        self.c = (g * g) if self.c is None else self.b2 * self.c + (1 - self.b2) * (g * g)
        mh = self.m / (1 - self.b1 ** self.t)
        ch = self.c / (1 - self.b2 ** self.t)
        return p - self.lr * mh / (np.sqrt(ch) + self.eps)


def val(A: np.ndarray, p: np.ndarray) -> float:
    if p is None or not np.all(np.isfinite(p)):
        return float("inf")
    return float(0.5 * p @ A @ p)


def run(A: np.ndarray, opt, p0=P0, iters: int = ITERS) -> np.ndarray:
    path = [np.asarray(p0, dtype=float)]
    p = path[0].copy()
    for _ in range(iters):
        p = opt.step(p, A @ p)                      # ∇f = Ap
        path.append(p if np.all(np.isfinite(p)) else np.array([np.nan, np.nan]))
        if not np.all(np.isfinite(p)):
            break
    return np.stack(path)


def oscillates(path: np.ndarray) -> bool:
    """좌표에 부호 반전이 있었는가 — '진동' 라벨은 관측으로만 붙인다."""
    fin = path[np.isfinite(path).all(axis=1)]
    if len(fin) < 4:
        return False
    d = np.diff(fin, axis=0)
    return bool(np.any(d[:-1] * d[1:] < 0))


def verdict(f0: float, f1: float, path: np.ndarray) -> str:
    if not np.isfinite(f1) or f1 > f0:
        return "발산"
    if f1 < F_CONV:
        return "수렴"
    return "진동" if oscillates(path) else "정체(천천히 감소)"


def main() -> dict:
    rows, paths = [], {}
    factories = [("vanilla", VanillaGD, 2.0), ("momentum", Momentum, 2.0 * (1 + BETA)),
                 ("adam(lr/10)", lambda lr: Adam(lr / 10.0), None)]
    for fname, A in FUNCS.items():
        hess_max = HESS[fname]
        for oname, mk, coef in factories:
            for lr in LRS:
                path = run(A, mk(lr))
                f0, f1 = val(A, P0), val(A, path[-1])
                radius = float(np.linalg.norm(path[-1])) if np.isfinite(path[-1]).all() else float("inf")
                rows.append({"함수": fname, "옵티": oname, "lr": lr, "f_시작": round(f0, 2),
                             "f_최종": (f"{f1:.3e}" if np.isfinite(f1) else "inf"),
                             "반경_최종": (f"{radius:.3e}" if np.isfinite(radius) else "inf"),
                             "판정": verdict(f0, f1, path),
                             "원문4-2_기준(반경≤0.1)": bool(radius <= R_SPEC),
                             "이론상한": (round(coef / hess_max, 6) if coef else None),
                             "이론상한_공식": (VanillaGD.upper if oname == "vanilla" else
                                               Momentum.upper if oname == "momentum" else Adam.upper)})
                paths[(fname, oname, lr)] = path

    conv = {}
    for fname, A in FUNCS.items():
        hess_max = HESS[fname]
        for oname, _, coef in factories:
            ok = [r["lr"] for r in rows if r["함수"] == fname and r["옵티"] == oname and r["판정"] == "수렴"]
            ok_spec = [r["lr"] for r in rows if r["함수"] == fname and r["옵티"] == oname
                       and r["원문4-2_기준(반경≤0.1)"]]
            conv[f"{fname} / {oname}"] = {"수렴_lr(자체기준 f<1e-8)": ok,
                                          "측정_상한(자체기준)": (max(ok) if ok else None),
                                          "통과_lr(원문기준 반경≤0.1)": ok_spec,
                                          "측정_상한(원문기준)": (max(ok_spec) if ok_spec else None),
                                          "이론_상한": (round(coef / hess_max, 6) if coef else None),
                                          "이론_공식": (VanillaGD.upper if oname == "vanilla" else
                                                        Momentum.upper if oname == "momentum" else Adam.upper)}

    # 원문 4-4 대조: "lr=0.5 이상에서 발산"은 어느 함수에서 성립하는가
    spec44 = []
    for lr in (0.5, 0.9, 1.0, 1.5):
        rec = {"lr": lr}
        for fname in FUNCS:
            for oname in ("vanilla", "momentum"):
                r = next(r for r in rows if r["함수"] == fname and r["옵티"] == oname and r["lr"] == lr)
                rec[f"{fname.split(':')[0]}_{oname}"] = f"{r['f_최종']}({r['판정']})"
        spec44.append(rec)

    # 그림 4: 경로를 같은 축에 겹친다(미션 문장: "하나의 Figure에").
    # 2×2 로 그리는 이유: 리포트가 주장하는 조건은 lr=0.15(원형은 수렴·타원은 발산/모멘텀 생존)인데
    # lr=0.05 만 그림에 담으면 '무난한 경우'만 보여 주장과 그림이 어긋난다.
    fig, axs = plt.subplots(2, 2, figsize=(9.4, 7.6))
    for row, (fname, A) in enumerate(FUNCS.items()):
        xs = np.linspace(-4, 4, 120); ys = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(xs, ys)
        Z = 0.5 * (A[0, 0] * X ** 2 + A[1, 1] * Y ** 2)
        for col, lr in enumerate((0.05, 0.15)):
            ax = axs[row][col]
            ax.contour(X, Y, Z, levels=np.logspace(-2, 3, 12), colors="0.75", linewidths=.7)
            for oname, mk, style in [("vanilla", VanillaGD, "--o"), ("momentum", Momentum, "-s")]:
                path = run(A, mk(lr))
                shown = path[np.isfinite(path).all(axis=1)]
                shown = shown[np.abs(shown).max(axis=1) <= 6.0]     # 축 밖(발산)은 잘라내고 그린다
                ax.plot(shown[:, 0], shown[:, 1], style, ms=2.4, lw=1.0, label=f"{oname} {len(shown)}/{len(path)}점")
            th = np.linspace(0, 2 * np.pi, 200)                      # 원문 4-2 기준(반경 0.1)을 그림에
            ax.plot(R_SPEC * np.cos(th), R_SPEC * np.sin(th), "g-", lw=1.2, label="원문 기준 반경 0.1")
            ax.plot(P0[0], P0[1], "k*", ms=10, label="시작점")
            van = next(r for r in rows if r["함수"] == fname and r["옵티"] == "vanilla" and r["lr"] == lr)
            mom = next(r for r in rows if r["함수"] == fname and r["옵티"] == "momentum" and r["lr"] == lr)
            ax.set_title(f"{fname} · lr={lr}\nvanilla {van['f_최종']}({van['판정']}) / "
                         f"momentum {mom['f_최종']}({mom['판정']})", fontsize=8)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal"); ax.legend(fontsize=6)
    save(fig, "04_optimizer_paths.png")

    # 그림 5: lr 스윕 — 발산 경계를 함수별로 나란히(점선은 **vanilla** 이론 상한 2/λmax)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (fname, A) in zip(axes, FUNCS.items()):
        for oname, color in [("vanilla", "#c0392b"), ("momentum", "#2980b9"), ("adam(lr/10)", "#27ae60")]:
            ys_ = [float(r["f_최종"]) if r["f_최종"] != "inf" else np.nan
                   for r in rows if r["함수"] == fname and r["옵티"] == oname]
            xs_ = [r["lr"] for r in rows if r["함수"] == fname and r["옵티"] == oname]
            ax.semilogy(xs_, np.clip(ys_, 1e-16, None), "o-", ms=3.4, lw=.9, color=color, label=oname)
        hess_max = HESS[fname]
        for coef, ls, lab in [(2.0, ":", "2/λmax(H) (vanilla)"),
                              (2.0 * (1 + BETA), "-.", "2(1+β)/λmax(H) (heavy-ball 안정 한계)")]:
            up = coef / hess_max
            ax.axvline(up, ls=ls, c="k", lw=1.1)
            ax.text(up * 1.05, 1e5 if coef == 2.0 else 1e2, f"{lab}={up:.3g}", fontsize=6.5, rotation=90, va="bottom")
        ax.set_title(fname); ax.set_xlabel("learning rate"); ax.set_ylabel("|f| 최종(로그)")
        ax.grid(alpha=.25)
        ax.set_xlim(0, 2.05)   # 원의 안정 한계 1.9 가 스윕 범위(최대 1.5) 너머라도 보이도록
    axes[0].legend(fontsize=7)
    save(fig, "05_lr_sweep.png")

    # '수렴_lr 없음'이 모멘텀의 패배를 뜻하는지 확인: 반복수를 늘려 다시 잰다(판정 기준이 비교 대상이 된다)
    longrun = []
    for fname, A in FUNCS.items():
        for lr in (0.05, 0.15):
            rec = {"함수": fname, "lr": lr}
            for it in (100, 1000, 10000):
                rec[f"mom_{it}"] = f"{val(A, run(A, Momentum(lr), iters=it)[-1]):.3e}"
                rec[f"van_{it}"] = f"{val(A, run(A, VanillaGD(lr), iters=it)[-1]):.3e}"
            longrun.append(rec)

    v_tae = [r for r in rows if r["함수"] == "타원: x^2 + 10y^2" and r["옵티"] == "vanilla" and r["lr"] == 0.15]
    m_tae = [r for r in rows if r["함수"] == "타원: x^2 + 10y^2" and r["옵티"] == "momentum" and r["lr"] == 0.15]
    c_van = [r for r in rows if r["함수"] == "원: x^2 + y^2" and r["옵티"] == "vanilla" and r["lr"] == 0.1]
    spec42 = {"원문": "Vanilla GD, f=x²+y², lr=0.1, 100회 → 반경 0.1 이내",
              "반경_최종": c_van[0]["반경_최종"], "f_최종": c_van[0]["f_최종"],
              "gate_원문4-2": c_van[0]["원문4-2_기준(반경≤0.1)"]}
    out = {"표": rows, "반복수_민감도": longrun, "수렴_상한": conv,
           "gate_원문4-2": spec42,
           "원문4-4_대조(lr≥0.5 발산)": {
               "원문": "lr=0.5 이상에서 최적화 경로가 발산함을 그래프로 확인한다",
               "표": spec44,
               "판정": ("**원(x²+y²)에서는 성립하지 않는다.** vanilla lr=0.5 → "
                        f"{next(r['f_최종'] for r in rows if r['함수'].startswith('원') and r['옵티'] == 'vanilla' and r['lr'] == 0.5)}"
                        "(발산 아님), momentum lr=0.5 → "
                        f"{next(r['f_최종'] for r in rows if r['함수'].startswith('원') and r['옵티'] == 'momentum' and r['lr'] == 0.5)}. "
                        "이론 상한이 원은 1.0, 타원은 0.1 이므로 'lr≥0.5 발산'은 **타원에서만** 참이다. "
                        "그림 05 의 두 점선(2/λmax, 2(1+β)/λmax)이 그 경계다")},
           "핵심_실측": {"타원_vanilla_lr0.15": v_tae[0]["f_최종"], "타원_momentum_lr0.15": m_tae[0]["f_최종"],
                        "원_vanilla_측정상한(자체기준)": conv["원: x^2 + y^2 / vanilla"]["측정_상한(자체기준)"],
                        "원_vanilla_측정상한(원문기준)": conv["원: x^2 + y^2 / vanilla"]["측정_상한(원문기준)"],
                        "원_이론상한": conv["원: x^2 + y^2 / vanilla"]["이론_상한"],
                        "타원_vanilla_측정상한(자체기준)": conv["타원: x^2 + 10y^2 / vanilla"]["측정_상한(자체기준)"],
                        "타원_vanilla_측정상한(원문기준)": conv["타원: x^2 + 10y^2 / vanilla"]["측정_상한(원문기준)"],
                        "타원_이론상한": conv["타원: x^2 + 10y^2 / vanilla"]["이론_상한"],
                        "타원_momentum_이론상한": conv["타원: x^2 + 10y^2 / momentum"]["이론_상한"]},
           "이론_상한_정의": {"vanilla": "lr < 2/λmax(H) — 원(H=2I) 1.0, 타원(H=diag(2,20)) 0.1",
                              "momentum(heavy-ball v←βv+g)": "발산 경계 = 안정 한계 lr < 2(1+β)/λmax(H) — 원 **1.9**, 타원 **0.19** "
                                                              "(실측: 타원 0.15 안정 → 0.2 발산, 경계 ∈(0.15,0.2] 로 일치). "
                                                              "실효 lr = lr/(1−β) = 10·lr 휴리스틱의 2(1−β)/λmax(H) = 원 0.1·타원 0.01 은 "
                                                              "'GD 동급 수렴속도' 상한(100회 내 f<1e-8 인 lr 이 없는 것과 정합)이지 발산 경계가 아니다",
                              "adam": "해석 상한 없음(적응형) — 측정만 싣는다",
                              "주의": "초판은 세 옵티마이저 모두에 2/λmax 를 적었고(1차 감사 지적), 1차 정정은 모멘텀에 "
                                      "실효lr 상한 2(1−β)/λmax 를 발산 경계로 잘못 넣었다 → '측정(원문기준) 1.5·0.15 > 이론 0.1·0.01' "
                                      "모순이 생김. 2차 감사에서 상태행렬 스펙트럼반경으로 안정 한계 2(1+β)/λmax 를 유도해 재 정정했다. "
                                      "그리고 코드 안 λmax 는 eigvalsh(A)[-1]=2·λmax(H) 였다 → 표기를 헤세 기준으로 통일"},
           "주의": (f"최종값은 {ITERS}회 반복 후이며 '수렴' 자체기준은 f<{F_CONV:g}(원문 기준은 반경 ≤{R_SPEC}). "
                    "기준을 바꾸면 상한이 같이 움직이므로 lr 단독 주장 금지(함수·기준·반복수를 함께 적는다). "
                    "'진동'은 좌표의 부호 반전이 관측된 경우에만 붙이고, 감소 중이나 기준 미달이면 '정체'로 쓴다. "
                    "모멘텀은 heavy-ball 변형: 발산 경계는 안정 한계 2(1+β)/λmax(H)(원 1.9·타원 0.19)이고, "
                    "실효 학습률 lr/(1−β)=10·lr 은 수렴속도 문제다(GD 동급 상한 2(1−β)/λmax=원 0.1·타원 0.01)"),
           "조건": f"시작점 {list(P0)}, 반복 {ITERS}, float64, 모멘텀 β={BETA}(heavy-ball), Adam lr=lr/10"}
    return write("optimizer.json", out)


if __name__ == "__main__":
    d = main()
    print(d["gate_원문4-2"])
    print(d["핵심_실측"])
    print(d["원문4-4_대조(lr≥0.5 발산)"]["판정"])
    print("lr 스윕 반복수 민감도:")
    for r in d["반복수_민감도"]:
        print("  ", r)
    for r in d["표"][:6]:
        print(r)
