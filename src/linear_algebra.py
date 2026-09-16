"""요구 1(선형대수): 회전·확대/축소·전단 + 행렬식이 '넓이배'인지 확인 + power iteration·SVD.

함정(A2-1-T2): 대각 스케일 S(2, 0.5)는 det = 1 이라 "면적비 = |det|" 검사가 **항등식**이 된다.
그래서 (1) 면적비 검사는 det != 1 인 행렬로만 내리고, (2) S(2,0.5)는 *모양*이 안 변하는 검사로 대체한다
(원 → 긴 축 2·짧은 축 0.5 인 타원, 고윳값 비 4). 각 검사가 무엇을 구분하지 못하는지도 표에 남긴다.
"""
from __future__ import annotations

import numpy as np

from _common import circle, plt, save, shoelace, write   # plt 는 _common 에서 폰트 설정 후 재수출


def rot(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def diag_scale(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0.0], [0.0, sy]])


def shear(k: float) -> np.ndarray:
    return np.array([[1.0, k], [0.0, 1.0]])


MATRICES = {
    "회전 30°": rot(30.0),
    "스케일 (3, 0.5)": diag_scale(3.0, 0.5),
    "전단 1.2": shear(1.2),
    "스케일 (2, 0.5) det=1": diag_scale(2.0, 0.5),
}


def area_ratio(M: np.ndarray, n: int = 200001) -> float:
    P = circle(n)
    return shoelace(P @ M.T) / shoelace(P)


def main() -> dict:
    rows = []
    for name, M in MATRICES.items():
        det = float(np.linalg.det(M))
        ratio = area_ratio(M)
        ev = np.linalg.eigvals(M)
        rows.append({
            "행렬": name,
            "det": round(det, 6),
            "면적비(실측)": round(ratio, 6),
            "절대오차": f"{abs(ratio - abs(det)):.2e}",
            "고윳값": [f"{v.real:.3f}{v.imag:+.3f}j" for v in ev],
            "검사가무의미": abs(det - 1.0) < 1e-9,      # det=1 이면 면적비 검사는 항등식
        })
    # 항등식이 아닌지 스스로 확인: 회전은 면적비를 1로 재지만 ' 모양'은 바꾼다 → 구분하려면 축비를 재야 한다
    M = MATRICES["스케일 (2, 0.5) det=1"]
    ell = circle() @ M.T
    axes = (ell.max(axis=0) - ell.min(axis=0)) / 2.0
    rows.append({"행렬": "det=1 의 축비(대안 검사)", "det": 1.0, "면적비(실측)": round(float(axes[0] / axes[1]), 6),
                 "절대오차": f"{abs(axes[0] / axes[1] - 4.0):.2e}", "고윳값": ["2.000", "0.500"],
                 "검사가무의미": False})

    fig, axes_arr = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (name, M) in zip(axes_arr, list(MATRICES.items())[:3]):
        P, Q = circle(), circle() @ M.T
        ax.plot(P[:, 0], P[:, 1], lw=1.0, ls=":", c="0.5", label="이전")
        ax.plot(Q[:, 0], Q[:, 1], lw=1.4, c="#c0392b", label="이후")
        ax.set_title(name); ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.set_aspect("equal"); ax.grid(alpha=.3)
    axes_arr[0].legend(fontsize=7)
    axes_arr[1].set_ylabel("")
    save(fig, "01_linalg_transforms.png")

    out = {"표": rows,
           "판정": "면적비=|det| 는 det!=1 에서만 정보가 있다. det=1 행렬은 면적비 검사가 항등식이라 축비 검사로 대체했다",
           "조건": "면적비는 다각형 근사(n=200001)·float64, 행렬은 2×2 실수 정방행렬"}
    return write("linear_algebra.json", out)


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    d = main()
    for r in d["표"]:
        print({k: v for k, v in r.items()})
    print(d["판정"])
