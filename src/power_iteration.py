"""요구 1-2(선형대수): 거듭제곱 반복(power iteration) — 수렴 속도는 |λ2/λ1| 이 결정한다.

A = Q diag(λ1, λ2) Qᵗ 로 비를 만들고, v0 는 λ1 고유벡터와 45° 어긋나게 둔다.

감사(2026-09-27) 반영 3건:
1. 원문 1-6 은 **np.linalg.eig** 와의 비교를 지정한다 → eig 비교 열을 추가(eigvalsh 는 대조로 병기).
2. '이론 예측 반복수'는 수렴 판정 기준과 같은 양으로 재야 한다. 이 구현의 기준은 **Rayleigh quotient 의
   상대변화 ≤1e-12** 이고 그 오차는 (λ2/λ1)^(2k) 로 준다 → 예측은 ln(tol)/ln(r²).
   이전 공식 ln(tol)/ln(r) 은 고유벡터 오차 기준이라 실측(105/19/7)의 2배를 예측했다(262/39/12).
3. v0 가 λ1 고유벡터에 직교인 행은 **성공이 아니라 실패 사례**다(λ_max 대신 λ2 를 반환) → 판정 열을 분리.
   또한 원문 1-5/1-6 이 요구하는 **시각화**(그림 08: Rayleigh 오차 수렴 곡선)를 추가한다.
"""
from __future__ import annotations

import numpy as np

from _common import plt, save, write

TOL = 1e-12


def power_iteration(A: np.ndarray, v0: np.ndarray | None = None, iters: int = 5000, tol: float = TOL,
                    trace: bool = False):
    v = np.array([1.0, 1.0]) if v0 is None else np.asarray(v0, dtype=float)
    v = v / np.linalg.norm(v)
    lam = float(v @ A @ v)
    hist = [lam]
    for k in range(1, iters + 1):
        w = A @ v
        nv = float(np.linalg.norm(w))
        if nv == 0.0:
            return (float("nan"), v, k, hist) if trace else (float("nan"), v, k)
        v = w / nv
        new = float(v @ A @ v)
        hist.append(new)
        if abs(new - lam) <= tol * max(1.0, abs(new)):
            return (new, v, k, hist) if trace else (new, v, k)
        lam = new
    return (lam, v, iters, hist) if trace else (lam, v, iters)


def make(ratio: float, lam1: float = 100.0) -> np.ndarray:
    th = np.deg2rad(23.0)
    Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return Q @ np.diag([lam1, lam1 * ratio]) @ Q.T


def predicted_iters(r: float, tol: float = TOL) -> int:
    """Rayleigh quotient 상대오차가 tol 아래로 오는 횟수: err_k ~ r^(2k) → k ≥ ln(tol)/ln(r²)."""
    if not (0 < r < 1):
        return 0
    return int(np.ceil(np.log(tol) / np.log(r * r)))


def main() -> dict:
    rows, curves = [], {}
    cases = [("λ2/λ1 = 0.9 (느림)", 0.9), ("λ2/λ1 = 0.5", 0.5), ("λ2/λ1 = 0.1 (빠름)", 0.1)]
    for label, r in cases:
        A = make(r)
        lam, v, it, hist = power_iteration(A, trace=True)
        w_eig, V_eig = np.linalg.eig(A)                 # 원문 1-6 이 지정한 함수
        lam_eig = float(np.max(np.abs(w_eig.real)))
        lam_sh = float(np.linalg.eigvalsh(A)[-1])       # 대칭행렬 대조(수치 동일해야 정상)
        err = abs(lam - lam_eig) / abs(lam_eig)
        rows.append({"case": label, "λ_실측": round(lam, 6),
                     "λ_참값_eig": round(lam_eig, 6), "λ_참값_eigvalsh": round(lam_sh, 6),
                     "상대오차%(vs eig)": f"{err * 100:.3e}",
                     "gate_원문1-6(≤5%)": bool(err <= 0.05),
                     "반복수": it, "이론_예측반복_Rayleigh": predicted_iters(r),
                     "이전공식_ln(tol)/ln(r)": int(np.log(TOL) / np.log(r)),
                     "v0": "[1,1] (λ1 고유벡터와 23°+45° 어긋남)",
                     "판정": "λ_max 수렴" if err <= 0.05 else "실패"})
        curves[label] = (r, np.array(hist), lam_eig, it)

    # v0 가 최대 고유벡터에 직교하는 경우 — **실패 사례**(λ_max 를 못 찾는다)
    A = make(0.9)
    w, V = np.linalg.eigh(A)
    lam2, v2, it2, hist2 = power_iteration(A, v0=V[:, 0], trace=True)
    w_eig2, _ = np.linalg.eig(A)
    lam_max = float(np.max(np.abs(w_eig2.real)))
    err2 = abs(lam2 - lam_max) / lam_max
    rows.append({"case": "v0 ⊥ 최대 고유벡터 (λ2 고유벡터로 시작)", "λ_실측": round(lam2, 6),
                 "λ_참값_eig": round(lam_max, 6), "λ_참값_eigvalsh": round(lam_max, 6),
                 "상대오차%(vs eig)": f"{err2 * 100:.3e}", "gate_원문1-6(≤5%)": bool(err2 <= 0.05),
                 "반복수": it2, "이론_예측반복_Rayleigh": None, "이전공식_ln(tol)/ln(r)": None,
                 "v0": "λ2 고유벡터 (eigh 로 구한 V[:,0])",
                 "판정": ("**실패 — λ_max 대신 λ2 를 반환.** v0 에 λ1 고유벡터 성분이 0이면 power iteration 은 "
                          "λ1 에 도달하지 못한다. 반복수 1·오차 0 은 'λ2 기준으로는 수렴'일 뿐 성공이 아니다 "
                          "(Rayleigh 변화가 첫 단계부터 0이라 즉시 정지). 이전 버전은 이 행을 상대오차 1.6e-14% 로 "
                          "성공처럼 기록했다")})

    # 그림 08: 원문 1-5/1-6 의 시각화 — Rayleigh 오차 수렴 곡선 + 실측 정지점 + 이론 예측
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for label, (r, hist, lam_true, it) in curves.items():
        err = np.abs(np.array(hist) - lam_true) / abs(lam_true)
        ks = np.arange(len(err))
        ax.semilogy(ks, np.clip(err, 1e-17, None), "o-", ms=3.0, lw=1.0,
                    label=f"r={r:g} — 실측 {it}회 / 예측 {predicted_iters(r)}회")
        ax.axvline(it, ls=":", lw=.8, color=ax.lines[-1].get_color())
    ax.axhline(TOL, ls="--", c="k", lw=1.0)
    ax.text(0.6, TOL * 2.2, f"수렴 기준 tol={TOL:g}", fontsize=7)
    ax.set_title("power iteration 수렴: Rayleigh quotient 상대오차 (log)")
    ax.set_xlabel("반복 k"); ax.set_ylabel("|λ_k − λ_max| / λ_max")
    ax.legend(fontsize=7); ax.grid(alpha=.25, which="both")
    save(fig, "08_power_iteration.png")

    iters_list = [r["반복수"] for r in rows[:3]]
    out = {"표": rows,
           "gate_원문1-6": {"조건": "직접 계산한 λ vs np.linalg.eig 최대고유값 상대오차 ≤5%",
                            "최대_상대오차%": f"{max(float(r['상대오차%(vs eig)']) for r in rows[:3]):.3e}",
                            "통과": all(r["gate_원문1-6(≤5%)"] for r in rows[:3]),
                            "eigvalsh_대조": "대칭행렬이라 eig 와 수치 동일(차이 1e-13 이하) — 원문 지정 함수는 eig"},
           "판정": ("반복수가 비에 따라 " + " → ".join(map(str, iters_list)) + " 로 단조 감소. "
                    "이론 예측 ln(tol)/ln(r²)=" + "/".join(str(predicted_iters(r)) for _, r in cases) +
                    " 과 실측의 차이는 초기 성분의 크기와 '변화량 기준 정지' 때문이며 자릿수가 맞는다. "
                    "이전 공식 ln(tol)/ln(r) 은 2배를 예측했다(고유벡터 오차 기준이라 Rayleigh 기준과 다르다). "
                    "마지막 행은 **실패 사례**로 남긴다 — 직교 초기벡터는 λ_max 를 찾지 못한다"),
           "조건": (f"A = Q(23°) diag(100, 100r) Qᵗ, 대칭 2×2, float64, 수렴 기준 |Rayleigh 변화| ≤ {TOL:g}(상대), "
                    "v0 는 행별로 표에 명시(기본 [1,1]), 그림 08 은 같은 실행의 곡선"),
           "그림": "figures/08_power_iteration.png"}
    return write("power_iteration.json", out)


if __name__ == "__main__":
    d = main()
    for r in d["표"]:
        print(r)
    print(d["gate_원문1-6"])
    print(d["판정"])
