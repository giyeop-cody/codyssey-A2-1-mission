"""요구 3(역전파): 2-2-1 고정 예제를 **손으로** 유도한 값과 NumPy·수치미분이 4자리에서 일치하는지.

규칙(하나 정해 노트북 전체에서 유지한다 — 뒤집으면 shape 이 어긋난다):
    z1 = W1 @ x + b1   (W1: (2,2), x: (2,))   ⇒  ∂L/∂W1 = outer(dz1, x) : (2,2)
    z2 = W2 @ a1 + b2  (W2: (2,))             ⇒  ∂L/∂W2 = a1 * dz2        : (2,)

주의(실측으로 드러남): x=[1,0] 이므로 W1 의 **두 번째 열**은 매개변수를 어떻게 흔들어도 손실이 변하지 않는다
(기울기 0). '기울기 0 = 오동작'이 아니라 입력 결핍이다 → 검사에 '0이 아닌 항목이 존재'를 넣는다.
"""
from __future__ import annotations

import numpy as np

from _common import write

X = np.array([1.0, 0.0])
W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
B1 = np.zeros(2)
W2 = np.array([0.5, 0.6])
B2 = 0.0
EPS = 1e-5


def sigm(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def forward(x=X, W1=W1, B1=B1, W2=W2, B2=B2):
    z1 = W1 @ x + B1
    a1 = sigm(z1)
    z2 = float(W2 @ a1 + B2)
    return z1, a1, z2, float(sigm(z2))


def bce(yhat: float, y: float) -> float:
    return float(-(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)))


def backward(y: float, x=X, W1=W1, B1=B1, W2=W2, B2=B2):
    """원문 3-5 가 요구하는 6항(∂L/∂y_pred, ∂L/∂z2, ∂L/∂W2, ∂L/∂a1, ∂L/∂z1, ∂L/∂W1)을 **빠짐없이** 반환한다.

    이전 버전은 dy_pred·da1 을 중간변수로만 쓰고 반환하지 않아, 노트북 룰표에서도 두 항이 빠져 있었다
    (∂L/∂z₂ = ŷ−y 로 직행). 편향 기울기 db1·db2 도 함께 반환한다.
    """
    z1, a1, z2, yhat = forward(x, W1, B1, W2, B2)
    dy_pred = -(y / yhat) + (1.0 - y) / (1.0 - yhat)   # ∂L/∂ŷ  (BCE 를 ŷ 로 직접 미분)
    dz2 = dy_pred * yhat * (1.0 - yhat)                # = ŷ − y  (시그모이드+BCE 의 상쇄)
    dW2 = a1 * dz2
    db2 = float(dz2)
    da1 = W2 * dz2                                     # ∂L/∂a₁
    dz1 = da1 * a1 * (1 - a1)
    dW1 = np.outer(dz1, x)
    db1 = dz1.copy()
    return {"z1": z1, "a1": a1, "z2": z2, "yhat": yhat, "L": bce(yhat, y),
            "dy_pred": float(dy_pred), "dz2": dz2, "dW2": dW2, "da1": da1, "dz1": dz1,
            "dW1": dW1, "db1": db1, "db2": db2}


# 원문 3-5 필수 유도 6항 ↔ 이 구현의 키 ↔ shape (노트북 룰표와 같은 순서)
REQUIRED_DERIVATIONS = [
    ("∂L/∂y_pred", "dy_pred", "()", "-(y/ŷ) + (1-y)/(1-ŷ)"),
    ("∂L/∂z2", "dz2", "()", "∂L/∂y_pred · ŷ(1-ŷ) = ŷ - y"),
    ("∂L/∂W2", "dW2", "(2,)", "a1 · dz2"),
    ("∂L/∂a1", "da1", "(2,)", "W2 · dz2"),
    ("∂L/∂z1", "dz1", "(2,)", "∂L/∂a1 ⊙ a1(1-a1)"),
    ("∂L/∂W1", "dW1", "(2,2)", "outer(dz1, x)"),
]


def numeric_grad(y: float):
    """손실 함수를 매개변수별로 중심차분. 해석 기울기의 **독립적** 검증이다(같은 수식을 후미로 돌리면 항등식이 된다)."""
    def loss(W1_, B1_, W2_, B2_):
        # forward 시그니처는 (x, W1, B1, W2, B2) — 인자를 하나 밀어서 넘기면 모양이 맞아도 값이 꼬인다
        _, _, _, yh = forward(x=X, W1=W1_, B1=B1_, W2=W2_, B2=B2_)
        return bce(yh, y)

    g = {}
    g["dW1"] = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            Wp, Wm = W1.copy(), W1.copy()
            Wp[i, j] += EPS; Wm[i, j] -= EPS
            g["dW1"][i, j] = (loss(Wp, B1, W2, B2) - loss(Wm, B1, W2, B2)) / (2 * EPS)
    g["dW2"] = np.zeros_like(W2)
    for i in range(len(W2)):
        Wp, Wm = W2.copy(), W2.copy()
        Wp[i] += EPS; Wm[i] -= EPS
        g["dW2"][i] = (loss(W1, B1, Wp, B2) - loss(W1, B1, Wm, B2)) / (2 * EPS)
    # b1 은 '2×2 항등행렬의 행'을 그대로 더하면 섭동이 1.0 이 된다 → EPS 를 반드시 곱한다(실수해서 4.8e3 이 떴다)
    g["db1"] = np.array([(loss(W1, B1 + EPS * np.eye(2)[i], W2, B2) - loss(W1, B1 - EPS * np.eye(2)[i], W2, B2)) / (2 * EPS)
                         for i in range(2)])
    g["db2"] = float((loss(W1, B1, W2, B2 + EPS) - loss(W1, B1, W2, B2 - EPS)) / (2 * EPS))
    return g


def grad_errs(a, b, zero_tol: float = 1e-8) -> dict:
    """기울기 검사의 오차는 **정의를 명시**해야 한다(분모가 0이면 상대오차가 폭주한다 — A6-1 에서도 같은 함정).

    - '0이 아닌 항'은 상대오차 |a-b|/max(|a|,|b|)
    - '구조적으로 0인 항'(x=[1,0] 때문에 죽는 열)은 절대오차로만 판정한다
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = np.abs(a - b)
    nz = np.maximum(np.abs(a), np.abs(b)) > zero_tol
    return {"상대오차(0이_아닌_항)": float(np.max(d[nz] / np.maximum(np.abs(a), np.abs(b))[nz])) if nz.any() else 0.0,
            "절대오차(전체)": float(np.max(d)),
            "0항_개수": int((~nz).sum())}


def main() -> dict:
    hand = {1.0: backward(1.0), 0.0: backward(0.0)}
    per_y = {}
    for y in (1.0, 0.0):
        h = hand[y]
        n = numeric_grad(y)
        per_param = {k: grad_errs(h[k], n[k]) for k in ("dW1", "dW2", "db1", "db2")}
        worst_rel = max(v["상대오차(0이_아닌_항)"] for v in per_param.values())
        worst_abs = max(v["절대오차(전체)"] for v in per_param.values())
        per_y[str(y)] = {
            "L": round(float(h["L"]), 6), "yhat": round(float(h["yhat"]), 6),
            "dy_pred(∂L/∂y_pred)": round(float(h["dy_pred"]), 6), "dz2": round(float(h["dz2"]), 6),
            "da1(∂L/∂a₁)": [round(float(v), 6) for v in h["da1"]],
            "dz1": [round(float(v), 6) for v in h["dz1"]],
            "db1": [round(float(v), 6) for v in h["db1"]], "db2": round(float(h["db2"]), 6),
            "dW2": [round(float(v), 6) for v in h["dW2"]], "dW1": [round(float(v), 6) for v in h["dW1"].ravel()],
            "수치미분_dW1": [round(float(v), 6) for v in n["dW1"].ravel()],
            "오차표": {k: {kk: (f"{vv:.3e}" if isinstance(vv, float) else vv) for kk, vv in v.items()}
                      for k, v in per_param.items()},
            "최대상대오차_0제외": f"{worst_rel:.3e}", "최대절대오차": f"{worst_abs:.3e}",
            "gate_1e-7": bool(worst_rel <= 1e-7 and worst_abs <= 1e-8),
            "gate_4자리일치": bool(np.allclose(np.asarray(h["dW1"], dtype=float).ravel(),
                                            np.asarray(n["dW1"], dtype=float).ravel(), atol=5e-5)),
        }
    z1, a1, z2, yhat = forward()
    zero_cols = int(np.count_nonzero(~np.any(hand[1.0]["dW1"], axis=0)))
    out = {
        "고정예제": {"x": X.tolist(), "W1": W1.tolist(), "b1": B1.tolist(), "W2": W2.tolist(), "b2": B2,
                     "z1": [round(float(v), 6) for v in z1], "a1": [round(float(v), 6) for v in a1],
                     "z2": round(z2, 6), "yhat": round(yhat, 6)},
        "관례": "z1 = W1 @ x + b1 (열벡터 관례) → dW1 = outer(dz1, x). 뒤집은 관례(x @ W1)면 dW1 = outer(x, dz1) 로 전치된다",
        "원문3-5_필수유도": {
            "표": [{"항": s, "키": k, "shape": sh, "수식": f} for s, k, sh, f in REQUIRED_DERIVATIONS],
            "y_true_별_값": {yy: {k: (round(float(hh[k]), 6) if np.ndim(hh[k]) == 0
                                      else [round(float(v), 6) for v in np.ravel(hh[k])])
                                  for _, k, _, _ in REQUIRED_DERIVATIONS}
                             for yy, hh in hand.items()},
            "판정": "원문 3-5 의 6항을 모두 반환·기록한다(이전 버전은 ∂L/∂y_pred·∂L/∂a₁ 이 중간변수라 리포트에 없었다)",
        },
        "y_true_별": per_y,
        "y_true_미정의_대응": "미션 고정 예제는 y_true 를 적지 않았다 → 1.0/0.0 **양쪽** 손계산을 싣고, "
                              "리포트·노트북은 y_true=1.0 을 표준으로 삼는다(근거: 문서 01 결정 문단)",
        "기울기_0_열": {"개수": zero_cols, "해석": "x=[1,0] 때문. 오동작이 아니라 입력 결핍 → '0이 아닌 항이 있다' 검사가 필요하다"},
        "조건": f"float64, 수치미분 h={EPS:g}(중심차분), 시드 불필요(결정론적)",
    }
    return write("backprop.json", out)


if __name__ == "__main__":
    d = main()
    print("고정예제:", d["고정예제"])
    for y, v in d["y_true_별"].items():
        print(f"y={y}:", {k: v[k] for k in ("L", "yhat", "dz2", "dW2", "dW1", "최대상대오차_0제외", "최대절대오차",
                                            "gate_1e-7", "gate_4자리일치")})
    print("0 열:", d["기울기_0_열"])
