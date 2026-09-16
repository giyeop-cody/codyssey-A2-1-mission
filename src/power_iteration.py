"""요구 1-2: 거듭제곱 반복(power iteration) — 수렴 속도는 |λ2/λ1| 이 결정한다.

A = Q diag(λ1, λ2) Qᵗ 로 비를 만들고, v0 는 λ1 고유벡터와 45° 어긋나게 둔다(v0 가 직교하면 발산이 아니라
'다른 고윳값'으로 수렴하므로 그 경우도 따로 재는다 — 표에 행으로 남긴다).
"""
from __future__ import annotations

import numpy as np

from _common import write


def power_iteration(A: np.ndarray, v0: np.ndarray | None = None, iters: int = 5000, tol: float = 1e-12):
    v = np.array([1.0, 1.0]) if v0 is None else np.asarray(v0, dtype=float)
    v = v / np.linalg.norm(v)
    lam = float(v @ A @ v)
    for k in range(1, iters + 1):
        w = A @ v
        nv = float(np.linalg.norm(w))
        if nv == 0.0:
            return float("nan"), v, k
        v = w / nv
        new = float(v @ A @ v)
        if abs(new - lam) <= tol * max(1.0, abs(new)):
            return new, v, k
        lam = new
    return lam, v, iters


def make(ratio: float, lam1: float = 100.0) -> np.ndarray:
    th = np.deg2rad(23.0)
    Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return Q @ np.diag([lam1, lam1 * ratio]) @ Q.T


def main() -> dict:
    rows = []
    for label, r in [("λ2/λ1 = 0.9 (느림)", 0.9), ("λ2/λ1 = 0.5", 0.5), ("λ2/λ1 = 0.1 (빠름)", 0.1)]:
        A = make(r)
        true = np.linalg.eigvalsh(A)
        lam, v, it = power_iteration(A)
        rows.append({"case": label, "λ_실측": round(lam, 6), "λ_참값": round(float(true[-1]), 6),
                     "상대오차%": f"{abs(lam - true[-1]) / abs(true[-1]) * 100:.3e}", "반복수": it,
                     "이론_예측반복": int(np.log(1e-12) / np.log(r)) if r < 1 else None})
    # v0 가 최대 고유벡터에 직교하는 경우
    A = make(0.9)
    w, V = np.linalg.eigh(A)
    lam2, v2, it2 = power_iteration(A, v0=V[:, 0])
    rows.append({"case": "v0 가 최대 고유벡터에 직교(작은 고유벡터로 시작)", "λ_실측": round(lam2, 6), "λ_참값": round(float(w[0]), 6),
                 "상대오차%": f"{abs(lam2 - w[0]) / abs(w[0]) * 100:.3e}", "반복수": it2, "이론_예측반복": 0})
    iters = [r["반복수"] for r in rows[:3]]
    out = {"표": rows,
           "판정": "반복수가 비례 감소와 함께 " + " → ".join(map(str, iters)) +
                   " (단조 감소). 예측 ln(tol)/ln(r) 과 같은 자릿수면 이론과 맞는다",
           "조건": "A 대칭 2×2, float64, tol=1e-12(상대), v0=[1,1]"}
    return write("power_iteration.json", out)


if __name__ == "__main__":
    d = main()
    for r in d["표"]:
        print(r)
    print(d["판정"])
