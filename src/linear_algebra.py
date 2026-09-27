"""요구 1(선형대수): 회전·확대/축소·전단 + 행렬식이 '넓이배'인지 확인.

원문 1-1~1-4(감사 반영):
  1-1 회전 R(θ) — 변환 전후를 **하나의 Figure에 겹쳐**
  1-2 스케일 **S(2, 0.5)** — 타원 형태 출력 (이전 구현은 이 행렬을 그림에서 빼고 표에만 실었다 → 4축으로 복원)
  1-3 전단 Sh(k) — 기울어진 형태
  1-4 det(A) vs 변환 후 **면적 비율** 오차 **1% 이내** 확인 (상대오차% 열을 추가해 원문 조건을 직접 판정)

함정(A2-1-T2): 대각 스케일 S(2, 0.5)는 det = 1 이라 "면적비 = |det|" 검사가 **항등식**이 된다.
그래서 (1) 면적비 검사는 det != 1 인 행렬에서만 정보가 있고, (2) S(2,0.5)는 *모양*이 안 변하는 검사로
보완한다(원 → 긴 축 2·짧은 축 0.5 인 타원). 보완 검사의 값은 **별도 열(축비)** 에 담는다 —
이전 구현은 축비 4.0 을 '면적비(실측)' 열에 넣어 열 이름과 측정 정의가 어긋났었다.
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


# 원문 순서: 회전 R(θ) → 스케일 S(2,0.5) → 전단 Sh(k). 마지막 행은 대조용(det≠1).
MATRICES = {
    "회전 R(30°)": rot(30.0),
    "스케일 S(2, 0.5)  [원문 1-2]": diag_scale(2.0, 0.5),
    "전단 Sh(1.2)": shear(1.2),
    "스케일 (3, 0.5)  [det≠1 대조]": diag_scale(3.0, 0.5),
}
SPEC_SCALE = "스케일 S(2, 0.5)  [원문 1-2]"


def area_ratio(M: np.ndarray, n: int = 200001) -> float:
    P = circle(n)
    return shoelace(P @ M.T) / shoelace(P)


def axis_ratio(M: np.ndarray, n: int = 200001) -> float:
    """변환된 원의 장축/단축 비 — det=1 행렬에서 면적비를 대신할 '모양' 검사."""
    ell = circle(n) @ M.T
    axes = (ell.max(axis=0) - ell.min(axis=0)) / 2.0
    return float(max(axes) / min(axes))


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
            "상대오차%": f"{abs(ratio - abs(det)) / max(abs(det), 1e-300) * 100:.3e}",
            "gate_원문1-4(≤1%)": bool(abs(ratio - abs(det)) <= 0.01 * abs(det)),
            "고윳값": [f"{v.real:.3f}{v.imag:+.3f}j" for v in ev],
            "검사가무의미": abs(det - 1.0) < 1e-9,      # det=1 이면 면적비 검사는 항등식
        })
    # 항등식이 아닌지 스스로 확인: S(2,0.5) 는 면적비를 1로 재지만 '모양'은 바꾼다 → 구분하려면 축비를 재야 한다
    M = MATRICES[SPEC_SCALE]
    ar = axis_ratio(M)
    rows.append({"행렬": "S(2,0.5) 축비(대안 검사 — 면적비가 아님)", "det": 1.0,
                 "면적비(실측)": round(area_ratio(M), 6),
                 "축비(실측)": round(ar, 6),
                 "절대오차": f"{abs(ar - 4.0):.2e}", "상대오차%": f"{abs(ar - 4.0) / 4.0 * 100:.3e}",
                 "gate_원문1-4(≤1%)": None,
                 "고윳값": ["2.000", "0.500"], "검사가무의미": False})

    # 그림 01: 원문 1-1/1-2/1-3 을 **하나의 Figure**(서브플롯 4)에. 각 축에서 변환 전후를 겹친다.
    fig, axes_arr = plt.subplots(1, 4, figsize=(13.2, 3.5))
    for ax, (name, M) in zip(axes_arr, MATRICES.items()):
        P, Q = circle(), circle() @ M.T
        ax.plot(P[:, 0], P[:, 1], lw=1.0, ls=":", c="0.5", label="변환 전(단위 원)")
        ax.plot(Q[:, 0], Q[:, 1], lw=1.4, c="#c0392b", label="변환 후")
        ax.set_title(f"{name}\ndet={np.linalg.det(M):.3g} · 면적비={area_ratio(M):.4f}", fontsize=8)
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.set_aspect("equal"); ax.grid(alpha=.3)
    axes_arr[0].legend(fontsize=7)
    for a in axes_arr[1:]:
        a.set_ylabel("")
    save(fig, "01_linalg_transforms.png")

    out = {"표": rows,
           "gate_원문1-4": {"조건": "면적비 vs |det| 상대오차 ≤1%",
                            "최대_상대오차%": f"{max(float(r['상대오차%']) for r in rows[:4]):.3e}",
                            "통과": all(r["gate_원문1-4(≤1%)"] for r in rows[:4]),
                            "주의": "통과하지만 회전·전단·S(2,0.5) 는 det=1 이라 **항등식**이다(아래 판정). "
                                    "정보를 주는 행은 det=1.5 인 대조 행뿐이다"},
           "판정": ("면적비=|det| 는 det!=1 에서만 정보가 있다. det=1 행렬(회전·전단·S(2,0.5))은 면적비 검사가 "
                    "항등식이라 **축비 검사**로 보완했고, 그 값은 '축비(실측)' 열에 따로 담는다"
                    "(면적비 열에 섞지 않는다 — 열 이름이 곧 측정 정의다)"),
           "조건": "면적비·축비는 다각형 근사(n=200001)·float64, 행렬은 2×2 실수 정방행렬, "
                   "그림 01 은 1×4 서브플롯(각 축에서 변환 전후 겹침)"}
    return write("linear_algebra.json", out)


if __name__ == "__main__":
    d = main()
    for r in d["표"]:
        print(r)
    print(d["gate_원문1-4"])
    print(d["판정"])
