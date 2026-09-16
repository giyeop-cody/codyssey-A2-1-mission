"""요구 4(최적화): 경사하강 vs 모멘텀, 그리고 '수렴'을 주장할 때 붙어야 할 조건.

이차형식 f(p) = ½ pᵗA p (A 대각) 에서 GD는 **0 < lr < 2/λmax** 여야 발산하지 않는다.
원(A=diag(2,2), λmax=2)과 타원(A=diag(2,20), λmax=20)에서 상한이 1.0 vs 0.1 로 10배 차이 →
같은 lr=0.5 가 한쪽은 수렴, 한쪽은 폭발한다. 그래서 'lr 스윕'은 함수마다 따로 재야 의미가 있다.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import save, write

FUNCS = {"원: x^2 + y^2": np.diag([2.0, 2.0]), "타원: x^2 + 10y^2": np.diag([2.0, 20.0])}
LRS = [0.01, 0.05, 0.09, 0.095, 0.1, 0.15, 0.2, 0.4, 0.5, 0.9, 0.99, 1.0, 1.5]
P0 = np.array([-3.0, 2.0])
ITERS = 100


class VanillaGD:
    name = "vanilla"

    def __init__(self, lr: float):
        self.lr = lr

    def step(self, p, g):
        return p - self.lr * g


class Momentum:
    name = "momentum"

    def __init__(self, lr: float, beta: float = 0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, p, g):
        self.v = g.copy() if self.v is None else self.beta * self.v + g
        return p - self.lr * self.v


class Adam:
    """보너스(요구문에 없으나 맵의 보너스 1순위). lr 는 1/10 로 줄여 쓴다(스케일 차이 방지)."""
    name = "adam"

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


def main() -> dict:
    rows, paths = [], {}
    factories = [("vanilla", lambda lr: VanillaGD(lr)), ("momentum", lambda lr: Momentum(lr)),
                 ("adam(lr/10)", lambda lr: Adam(lr / 10.0))]
    for fname, A in FUNCS.items():
        lam_max = float(np.max(np.linalg.eigvalsh(A)))
        for oname, mk in factories:
            for lr in LRS:
                path = run(A, mk(lr))
                f0, f1 = val(A, P0), val(A, path[-1])
                rows.append({"함수": fname, "옵티": oname, "lr": lr, "f_시작": round(f0, 2),
                             "f_최종": (f"{f1:.3e}" if np.isfinite(f1) else "inf"),
                             "판정": ("발산" if (not np.isfinite(f1) or f1 > f0) else
                                      ("수렴" if f1 < 1e-8 else "미수렴(진동)")),
                             "이론상한_2/λmax": round(2 / lam_max, 4)})
                paths[(fname, oname, lr)] = path

    conv = {}
    for fname, A in FUNCS.items():
        for oname, _ in factories:
            ok = [r["lr"] for r in rows if r["함수"] == fname and r["옵티"] == oname and r["판정"] == "수렴"]
            conv[f"{fname} / {oname}"] = {"수렴_lr": ok,
                                          "측정_상한": (max(ok) if ok else None),
                                          "이론_상한": round(2 / float(np.max(np.linalg.eigvalsh(A))), 4)}

    # 그림 4: 경로를 같은 축에 겹친다(미션 문장: "하나의 Figure에")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, (fname, A) in zip(axes, FUNCS.items()):
        xs = np.linspace(-4, 4, 120)
        ys = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(xs, ys)
        Z = 0.5 * (A[0, 0] * X ** 2 + A[1, 1] * Y ** 2)
        ax.contour(X, Y, Z, levels=np.logspace(-2, 3, 12), colors="0.75", linewidths=.7)
        for oname, mk, style in [("vanilla", lambda lr: VanillaGD(lr), "--o"), ("momentum", lambda lr: Momentum(lr), "-s")]:
            p = run(A, mk(0.05))
            ax.plot(p[:, 0], p[:, 1], style, ms=2.6, lw=1.0, label=oname)
        ax.plot(P0[0], P0[1], "k*", ms=11, label="시작점")
        ax.set_title(f"{fname} (lr=0.05)")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        ax.legend(fontsize=7)
    save(fig, "04_optimizer_paths.png")

    # 그림 5: lr 스윕 — 발산 경계를 함수별로 나란히
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (fname, A) in zip(axes, FUNCS.items()):
        for oname, color in [("vanilla", "#c0392b"), ("momentum", "#2980b9"), ("adam(lr/10)", "#27ae60")]:
            ys_ = [float(r["f_최종"]) if r["f_최종"] != "inf" else np.nan
                   for r in rows if r["함수"] == fname and r["옵티"] == oname]
            xs_ = [r["lr"] for r in rows if r["함수"] == fname and r["옵티"] == oname]
            ax.semilogy(xs_, np.clip(ys_, 1e-16, None), "o-", ms=3.4, lw=.9, color=color, label=oname)
        up = 2 / float(np.max(np.linalg.eigvalsh(A)))
        ax.axvline(up, ls=":", c="k", lw=1.2)
        ax.text(up * 1.03, 1e5, f"2/λmax={up:.2f}", fontsize=7)
        ax.set_title(fname); ax.set_xlabel("learning rate"); ax.set_ylabel("|f| 최종(로그)")
        ax.grid(alpha=.25)
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
    out = {"표": rows, "반복수_민감도": longrun, "수렴_상한": conv,
           "핵심_실측": {"타원_vanilla_lr0.15": v_tae[0]["f_최종"], "타원_momentum_lr0.15": m_tae[0]["f_최종"],
                        "원_vanilla_측정상한": conv["원: x^2 + y^2 / vanilla"]["측정_상한"],
                        "원_이론상한": conv["원: x^2 + y^2 / vanilla"]["이론_상한"],
                        "타원_vanilla_측정상한": conv["타원: x^2 + 10y^2 / vanilla"]["측정_상한"],
                        "타원_이론상한": conv["타원: x^2 + 10y^2 / vanilla"]["이론_상한"]},
           "주의": "최종값은 100회 반복 후이며 '수렴' 기준은 f<1e-8. 기준을 바꾸면 상한이 같이 움직이므로 "
                  "lr 단독 주장 금지(함수·기준·반복수를 함께 적는다)",
           "조건": f"시작점 {list(P0)}, 반복 {ITERS}, float64, 모멘텀 β=0.9, Adam lr=lr/10"}
    return write("optimizer.json", out)


if __name__ == "__main__":
    d = main()
    print(d["핵심_실측"])
    print("lr 스윕 반복수 민감도:")
    for r in d["반복수_민감도"]:
        print("  ", r)
    for r in d["표"][:6]:
        print(r)
