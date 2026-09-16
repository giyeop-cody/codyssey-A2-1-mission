"""요구 5(확률): 손실 함수는 '선택'이 아니라 **가정에서 나온 결과**임을 수치로 확인한다.

- 정규 PDF(σ=1,3) / 베르누이 PMF(p=0.3,0.7) 그림 2종
- MSE = MLE: y = f(x) + ε(σ 고정) 일 때 로그우도 -N/2·log(2πσ²) - SSE/(2σ²) 의 최소점은 SSE 최소점과 **동일**
- CE = MLE: 베르누이 우도의 로그 = -(긍·log p + 음·log(1-p)) = BCE·N → 인자(argument)가 같은지 실측
- softmax 수치 안정성: logits 를 1000배 키우면 max 차감 없이 exp 오버플로(inf) → 차감 버전은 유한
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import save, write

RNG_SEED = 42


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def softmax(z: np.ndarray, stable: bool = True) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if stable:
        z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def main() -> dict:
    x = np.linspace(-8, 8, 400)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for s in (1.0, 3.0):
        ax.plot(x, normal_pdf(x, 0.0, s), lw=1.4, label=f"σ={s:g}")
        ax.axvspan(-s, s, alpha=.08)
    ax.set_title("정규 PDF 2종(σ=1 vs 3) · 같은 넓이 1, 높이만 다름")
    ax.set_xlabel("x"); ax.set_ylabel("밀도"); ax.legend(fontsize=7)
    save(fig, "06_normal_pdf.png")

    k = np.arange(0, 11)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for p in (0.3, 0.7):
        ax.bar(k + (0.18 if p > 0.5 else -0.18), p ** k * (1 - p) ** (10 - k), width=.34, label=f"p={p}")
    ax.set_title("베르누이(베르누이 시퀀스의 확률) PMF: n=10 중 k 번 성공")
    ax.set_xlabel("성공 횟수 k"); ax.set_ylabel("P(X=k)"); ax.legend(fontsize=7)
    save(fig, "07_bernoulli_pmf.png")

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
              "해석": f"CE(=BCE) 최소와 MLE 는 같은 p({mle_p:.4f}) 를 고른다(격자 0.001 이내 차이 0). "
                      f"그리고 깁스 부등식은 **참값 기준** CE {ce_pop:.4f} ≥ H {ent:.4f} 로 성립하고, "
                      f"표본 자체의 플러그인 엔트로피({nll_n:.4f})는 {nll_n - ent:+.4f} 만큼 **아래로** 작다 "
                      "→ '모델 손실이 엔트로피보다 크면 이상하다'는 문장은 표본 기준으로는 틀린다(플러그인 편향)"}

    # ── softmax 안정성 ─────────────────────────────────────────────────────────────────
    z = np.array([1.0, 2.0, 3.0])
    with np.errstate(over="ignore"):
        big = z * 1000.0
        naive = softmax(big, stable=False)
        safe = softmax(big, stable=True)
    small = softmax(z, stable=False)
    out_sm = {"원본_차이": f"{np.max(np.abs(softmax(z, False) - softmax(z, True))):.3e}",
              "1000배_max차감없음": str(naive.tolist()), "1000배_max차감있음": str(safe.tolist()),
              "차감버전_유한": bool(np.all(np.isfinite(safe))), "naive_유한": bool(np.all(np.isfinite(naive))),
              "해석": "logits ×1000 에서 차감 없으면 inf/inf → nan, 차감하면 유한하다. 그런데 원래 스케일에서는 "
                      "두 결과가 0 차이로 같다 → '안정화'는 큰 스케일에서 검증해야 의미가 있다(작은 스케일 검사는 항등식)"}

    out = {"정규PDF": {"σ=1 peak": round(float(normal_pdf(0, 0, 1)), 6), "σ=3 peak": round(float(normal_pdf(0, 0, 3)), 6),
                       "넓이_비": f"{np.trapezoid(normal_pdf(x,0,1), x):.6f} vs {np.trapezoid(normal_pdf(x,0,3), x):.6f}",
           "넓이_주의": "적분 구간 [-8,8] 이 유한해 σ=3 은 3σ 밖 0.27% 를 잘린다(0.992338). '넓이=1'을 주장하면 "
                        "구간을 함께 적어야 한다(계수에는 잣대 원칙)"},
           "MSE_MLE_동치": out_mse, "CE_MLE_동치": out_ce, "softmax": out_sm,
           "조건": f"시드 {RNG_SEED}(rng 기본값) · n=200 회귀 / 500 베르누이 · float64 · 격자 3001·4001",
           "주의": "표에 적힌 '차이'는 격자 해상도(0.001) 이내에서 0이다. 격자를 굵게 하면 0이 아니게 되니 "
                  "'완전 일치'라고 쓰지 않는다(계수에는 잣대 원칙)"}
    return write("probability.json", out)


if __name__ == "__main__":
    d = main()
    for k in ("정규PDF", "MSE_MLE_동치", "CE_MLE_동치", "softmax"):
        print(k, ":", d[k])
