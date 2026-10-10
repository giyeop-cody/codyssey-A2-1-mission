"""요구 5(확률): 손실 함수는 '선택'이 아니라 **가정에서 나온 결과**임을 수치로 확인한다.

- 정규 PDF: 원문 지정 **N(0,1) vs N(2,0.5)** (2026-09-27 감사 반영 — 이전 구현은 N(0,1) vs N(0,3) 이었다)
- 베르누이 PMF: 원문 지정 **B(0.3) vs B(0.7)**, k∈{0,1} 두 막대 · **합=1** 을 수치로 확인
  (이전 구현은 `p^k(1-p)^(10-k)`, k=0..10 — 이항계수가 없어 합이 0.647/0.249 로 1 이 아니었다.
   '베르누이' 라벨을 단 채 어떤 확률분포도 아니었으므로 정정하고, 배우려던 Binomial(10,p) 은 별도 항목으로 분리)
- MSE = MLE: y = f(x) + ε(σ 고정) 일 때 로그우도 -N/2·log(2πσ²) - SSE/(2σ²) 의 최소점은 SSE 최소점과 **동일**
- CE = MLE: 베르누이 우도의 로그 = -(긍·log p + 음·log(1-p)) = BCE·N → 인자(argument)가 같은지 실측
- softmax: 원문 5-3 **출력 합이 1.0 과 1e-6 이내** 인지 세 스케일에서 확인 + 수치 안정성(×1000) 검증
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import fig_to_png_b64, fig_to_png_bytes, save, write

RNG_SEED = 42

# 원문 5-1 이 지정한 두 정규분포
NORMALS = {"N(0,1)": (0.0, 1.0), "N(2,0.5)": (2.0, 0.5)}
# 원문 5-2 가 지정한 두 베르누이
BERNOULLIS = {"B(0.3)": 0.3, "B(0.7)": 0.7}


def normal_pdf(x, mu, sigma):
    """N(μ, σ) 의 PDF  φ(x) = exp(−(x−μ)²/(2σ²)) / (σ√(2π)).  sigma 는 **표준편차**(NumPy·SciPy scale 관례).

    원문 N(2,0.5) 를 σ=0.5 로 그린다(정점 1/(0.5√(2π)) = 0.797885). 교과서 N(μ,σ²) 관례면 σ=√0.5(정점 0.564190).
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def bernoulli_pmf(p: float) -> np.ndarray:
    """B(p) 의 PMF: k∈{0,1} → [1-p, p]. 합은 반드시 1."""
    return np.array([1.0 - p, p])


def binomial_pmf(n: int, p: float) -> np.ndarray:
    """Binomial(n,p): P(k) = C(n,k) p^k (1−p)^(n−k), k=0..n. (베르누이 = n=1 인 특수형 — 대조용)"""
    from math import comb
    return np.array([comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)])


def softmax(z: np.ndarray, stable: bool = True) -> np.ndarray:
    """softmax(z)_c = exp(z_c) / Σ_j exp(z_j).

    stable=True 면 max(z) 를 빼서 계산한다: exp(z_c − m)/Σexp(z_j − m) 는 분자·분모의 e^(−m) 이 약분돼 **수학적으로 같은 값**이고,
    exp 의 overflow(z×1000 → inf/inf = nan)만 막는다. 출력 합 = 1 (원문 5-3: |합 − 1| ≤ 1e-6).
    """
    z = np.asarray(z, dtype=float)
    if stable:
        z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def main() -> dict:
    # ── 그림 06: 원문 지정 N(0,1) vs N(2,0.5) 를 **하나의 Figure**(단일 축)에 겹친다 ──────────
    x = np.linspace(-8, 8, 1601)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for label, (mu, s) in NORMALS.items():
        ax.plot(x, normal_pdf(x, mu, s), lw=1.4, label=f"{label} (peak {normal_pdf(mu, mu, s):.4f})")
        ax.axvspan(mu - s, mu + s, alpha=.07)
    ax.set_title("원문 5-1: 정규 PDF N(0,1) vs N(2,0.5) — 중심과 폭이 둘 다 다르다")
    ax.set_xlabel("x"); ax.set_ylabel("밀도"); ax.legend(fontsize=7)
    save(fig, "06_normal_pdf.png")

    # ── 그림 07: 원문 지정 B(0.3) vs B(0.7) PMF (k=0,1) ────────────────────────────────────
    k = np.array([0, 1])
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, (label, p) in enumerate(BERNOULLIS.items()):
        pmf = bernoulli_pmf(p)
        ax.bar(k + (-0.18 if i == 0 else 0.18), pmf, width=.34, label=f"{label} (합 {pmf.sum():.6f})")
        for kk, vv in zip(k, pmf):
            ax.text(kk + (-0.18 if i == 0 else 0.18), vv + .012, f"{vv:.2f}", ha="center", fontsize=7)
    ax.set_xticks([0, 1]); ax.set_ylim(0, 0.85)
    ax.set_title("원문 5-2: 베르누이 PMF B(0.3) vs B(0.7) — 시행 1회, 합=1")
    ax.set_xlabel("성공 횟수 k (시행 1회 → k∈{0,1})"); ax.set_ylabel("P(X=k)"); ax.legend(fontsize=7)
    save(fig, "07_bernoulli_pmf.png")

    xw = np.linspace(-40, 40, 80001)
    norm_rows = {}
    trunc = {}
    for label, (mu, s) in NORMALS.items():
        i8 = float(np.trapezoid(normal_pdf(x, mu, s), x))
        i40 = float(np.trapezoid(normal_pdf(xw, mu, s), xw))
        norm_rows[label] = {"mu": mu, "sigma": s, "peak": round(float(normal_pdf(mu, mu, s)), 6),
                            "적분[-8,8]": round(i8, 6), "적분[-40,40]": round(i40, 6)}
        trunc[label] = 1.0 - i8
    bern_rows = {label: {"PMF_k0_k1": [round(float(v), 6) for v in bernoulli_pmf(p)],
                         "합": round(float(bernoulli_pmf(p).sum()), 12),
                         "합-1_절대오차": f"{abs(float(bernoulli_pmf(p).sum()) - 1.0):.3e}"}
                 for label, p in BERNOULLIS.items()}
    # 이전 구현이 '베르누이'라 부르던 것의 정체(감사 기록용 — 확률분포가 아니다)
    kk = np.arange(0, 11)
    old = {f"p={p}": round(float(np.sum(p ** kk * (1 - p) ** (10 - kk))), 6) for p in (0.3, 0.7)}
    binom = {f"Binomial(10,{p})": {"합": round(float(binomial_pmf(10, p).sum()), 12),
                                    "PMF_일부": [round(float(v), 6) for v in binomial_pmf(10, p)[[0, 3, 10]]]}
             for p in (0.3, 0.7)}

    # ── MSE = MLE (정규 오차 가정) ────────────────────────────────────────────────────
    rng = np.random.default_rng(RNG_SEED)
    n = 200
    xs = np.linspace(0, 5, n)
    f_true = 1.5 * xs + 2.0
    ys = f_true + rng.normal(0, 1.7, n)
    w = np.linspace(0.0, 3.0, 3001)
    pred = np.outer(w, xs) + 2.0
    sse = np.mean((pred - ys) ** 2, axis=1)
    sigma2 = np.var(ys - f_true, ddof=0)
    ll = -n / 2 * np.log(2 * np.pi * sigma2) - np.sum((pred - ys) ** 2, axis=1) / (2 * sigma2)
    mse_argmin, mle_argmax = float(w[np.argmin(sse)]), float(w[np.argmax(ll)])
    out_mse = {"기울기_MSE최소": round(mse_argmin, 6), "기울기_로그우도최대": round(mle_argmax, 6),
               "차이": f"{abs(mse_argmin - mle_argmax):.3e}", "σ²_참값추정": round(float(sigma2), 4),
               "해석": "σ 고정 시 두 목적함수는 상수배·상수합의 관계 → 인자가 같다(정규화 상수는 기울기에 무영향)"}

    # ── CE = MLE (베르누이) ────────────────────────────────────────────────────────────
    p_true = 0.35
    draws = (rng.random(500) < p_true).astype(float)
    ps = np.linspace(1e-6, 1 - 1e-6, 4001)
    # 후보 p(4001개) × 표본(500개) → 축을 맞춰 합산. (1-D @ 1-D 는 길이가 다르면 에러라 이 형태가 필요하다)
    LL = (np.log(ps)[:, None] * draws[None, :]).sum(axis=1) + \
         (np.log(1 - ps)[:, None] * (1 - draws)[None, :]).sum(axis=1)
    BCE = -(draws.mean() * np.log(ps) + (1 - draws.mean()) * np.log(1 - ps))
    mle_p, ce_p = float(ps[np.argmax(LL)]), float(ps[np.argmin(BCE)])
    nll_n = float(-np.max(LL) / len(draws))
    ent = float(-(p_true * np.log(p_true) + (1 - p_true) * np.log(1 - p_true)))
    # 깁스(교차엔트로피 ≥ 엔트로피)는 **모수 대비**로 성립한다. 표본의 플러그인 엔트로피는 아래로 왜곡된다.
    ce_pop = float(-(p_true * np.log(mle_p) + (1 - p_true) * np.log(1 - mle_p)))
    out_ce = {"로그우도_최대의_p": round(mle_p, 6), "BCE_최소의_p": round(ce_p, 6),
              "차이": f"{abs(mle_p - ce_p):.3e}", "표본_평균": round(float(draws.mean()), 6),
              "플러그인_엔트로피": round(nll_n, 6), "진_엔트로피": round(ent, 6),
              "참값대_ fitted_교차엔트로피": round(ce_pop, 6),
              "깁스_성립": bool(ce_pop >= ent - 1e-12),
              "플러그인_편향": f"{nll_n - ent:+.4f}",
              "해석": f"CE(=BCE) 최소와 MLE 는 같은 p({mle_p:.4f}) 를 고른다(같은 격자 Δp=0.00025 위에서 차이 0). "
                      f"그리고 깁스 부등식은 **참값 기준** CE {ce_pop:.4f} ≥ H {ent:.4f} 로 성립하고, "
                      f"표본 자체의 플러그인 엔트로피({nll_n:.4f})는 {nll_n - ent:+.4f} 만큼 **아래로** 작다 "
                      "→ 깁스(CE ≥ H)만 믿고 '학습 손실이 엔트로피보다 **작으면** 이상하다'고 쓰면 표본 기준으로는 틀린다(플러그인 편향). (2026-10-11 정정: 이전 문장은 '크면 이상하다'로 부등호 방향이 거꾸로였다)"}

    # ── softmax: 원문 5-3 (합 = 1.0 ± 1e-6) + 수치 안정성 ───────────────────────────────
    z = np.array([1.0, 2.0, 3.0])
    sum_cases = {}
    # over 뿐만 아니라 invalid 도 억누른다: inf/inf 에서 나는 RuntimeWarning 이 노트북 stderr 에
    # **절대경로**를 남기면 클론 위치마다 산출물이 달라진다(재현성 훼손).
    with np.errstate(over="ignore", invalid="ignore"):
        for label, zz in [("원본(1,2,3)", z), ("×1000", z * 1000.0), ("큰 음수(-500,0,500)", np.array([-500.0, 0.0, 500.0]))]:
            s = softmax(zz, stable=True)
            sum_cases[label] = {"출력": [round(float(v), 6) for v in s],
                                "합-1_절대오차": f"{abs(float(s.sum()) - 1.0):.3e}",
                                "gate_1e-6": bool(abs(float(s.sum()) - 1.0) <= 1e-6)}
        big = z * 1000.0
        naive = softmax(big, stable=False)
        safe = softmax(big, stable=True)
    worst = max(abs(float(softmax(zz, True).sum()) - 1.0)
                for zz in (z, z * 1000.0, np.array([-500.0, 0.0, 500.0])))
    out_sm = {"합=1_검증(원문5-3)": sum_cases,
              "최대_합-1_절대오차": f"{worst:.3e}",
              "gate_합=1_±1e-6": bool(worst <= 1e-6),
              "원본_차이": f"{np.max(np.abs(softmax(z, False) - softmax(z, True))):.3e}",
              "1000배_max차감없음": str(naive.tolist()), "1000배_max차감있음": str(safe.tolist()),
              "차감버전_유한": bool(np.all(np.isfinite(safe))), "naive_유한": bool(np.all(np.isfinite(naive))),
              "해석": "logits ×1000 에서 차감 없으면 inf/inf → nan, 차감하면 유한하다. 그런데 원래 스케일에서는 "
                      "두 결과가 0 차이로 같다 → '안정화'는 큰 스케일에서 검증해야 의미가 있다(작은 스케일 검사는 항등식). "
                      "원문 5-3 의 '합=1 ±1e-6' 은 세 스케일 모두 통과(최대 오차 위 값)"}

    out = {"정규PDF(원문5-1)": norm_rows,
           "넓이_주의": ("적분 구간 [-8,8] 이 유한하므로 잘림 1−∫ 를 재면 "
                        + ", ".join(f"{k} {v:.2e}" for k, v in trunc.items())
                        + " 로 둘 다 무시 가능(6자리 반올림에서 1.000000). 구간 의존성이 **보이는** 경우는 꼬리가 넓은 "
                          "분포뿐이다(예: 이전 구현이 쓰던 σ=3 은 [-8,8] 에서 0.992338). '넓이=1'을 주장하려면 구간을 함께 "
                          "적어야 한다(계수에는 잣대 원칙) — [-40,40] 열이 사실상 참값"),
           "베르누이PMF(원문5-2)": bern_rows,
           "베르누이PMF_정정_기록": {
               "이전_구현": "p^k (1-p)^(10-k), k=0..10 을 '베르누이 PMF' 라벨로 그림",
               "이전_구현_막대합": old,
               "왜_틀렸나": "이항계수 C(10,k) 가 없고 k∈{0,1} 밖의 항이 있다 → 합이 1 이 아니므로 확률분포가 아니다. "
                            "베르누이는 시행 1회라 막대가 k=0,1 두 개여야 한다",
               "분리_항목(학습용)": binom},
           "MSE_MLE_동치": out_mse, "CE_MLE_동치": out_ce, "softmax": out_sm,
           "조건": f"시드 {RNG_SEED}(rng 기본값) · n=200 회귀 / 500 베르누이 · float64 · 격자 3001·4001 · "
                   "정규 적분 격자 [-8,8] 1601점 / [-40,40] 80001점",
           "주의": "표에 적힌 '차이'는 격자 해상도(0.001) 이내에서 0이다. '완전 일치'라고 쓰지 않는다"
                  "(계수에는 잣대 원칙). 격자 해상도를 바꿔도 두 목적함수는 같은 격자 위에서 단조관계이므로 "
                  "argmin 이 갈린다는 주장은 실측 없이 쓰지 않는다"}
    return write("probability.json", out)


if __name__ == "__main__":
    d = main()
    for k in ("정규PDF(원문5-1)", "베르누이PMF(원문5-2)", "MSE_MLE_동치", "CE_MLE_동치", "softmax"):
        print(k, ":", d[k])
