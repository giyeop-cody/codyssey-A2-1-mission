"""역전파 노트북(notebooks/backprop_derivation.ipynb) 셀 정의 — build_notebook.py 가 import 한다.

2026-10-11 재검증 반영(원문 3-1~3-7 재대조):
- 이전 노트북은 수식 **결과표**만 있고 '출력층 → 입력층 단계별 유도'(3-2)가 없었다.
- 손계산은 순전파(ŷ·L)만 있었고, 역전파 6항의 손계산은 코드가 계산한 값이었다.
- '손계산 vs NumPy 4자리 일치'(3-7; 이전 문서 표기 3-8)는 실제로는 NumPy 해석기울기 vs 수치미분(gradcheck) 비교였다.
→ 유도(마크다운 LaTeX) · 손계산(소수 6자리로 반올림하며 계산기처럼 진행, 값은 마크다운에 고정) ·
  노트북 안 **독립 NumPy 구현** · 손값과 4자리 대조 · gradcheck 를 각각 분리된 절로 둔다.
"""

CELLS: list[tuple[str, str]] = []


def md(t: str) -> None:
    CELLS.append(("md", t))


def code(t: str) -> None:
    CELLS.append(("code", t))


md(r"""# 2층 신경망 역전파 — 수식 유도 · 손계산 · NumPy 검증

원문 요구(§4-3)와 이 노트북의 절 대응

| 원문 | 내용 | 절 |
|---|---|---|
| 3-1 | 2층 신경망(2→2→1) 순전파 수식 | §1 |
| 3-2 | 출력층 → 입력층 방향 연쇄 법칙, 단계별 역전파 유도 | §2 |
| 3-3 · 3-4 | 고정 예제 순전파 손계산 · 중간값 z1, a1, z2, y_pred 기록 | §3 |
| 3-5 | ∂L/∂y_pred, ∂L/∂z2, ∂L/∂W2, ∂L/∂a1, ∂L/∂z1, ∂L/∂W1 모두 유도 | §2(기호) · §4(숫자) |
| 3-6 | 각 단계 텐서 shape 표기 | §1~§4 모든 식 옆 |
| 3-7 | NumPy 로 동일 계산 → 손계산과 소수 4자리 일치 확인 | §5 |
| (추가) | 수치미분 gradcheck — 해석식 자체의 독립 검증 | §6 |
| (원문 결함) | y_true 공란 · 행렬 곱 관례 | §0 · §7 |

## §0. 정하고 시작하는 두 가지

1. **정답 y_true** — 원문 예제는 `정답: y_true =` 로 **값이 비어 있다**. 이 노트북은 **y_true = 1** 을 표준으로 계산하고,
   §7 에 y_true = 0 의 전 과정을 같이 싣는다(어느 쪽이 의도였든 채점 가능하게).
   1 을 표준으로 둔 이유는 관례(이진 분류 예제는 보통 양성 샘플로 시작)일 뿐, 원문에 근거가 있는 것은 아니다.
2. **곱셈 관례** — `W1 = [[0.1, 0.2], [0.3, 0.4]]` 를 **행 = 은닉 뉴런** 으로 읽어 $z_1 = W_1 x + b_1$ (열벡터 관례)을 쓴다.
   반대 관례 $z_1 = x W_1$ 이면 $z_1=[0.1, 0.2]$ 가 되어 ŷ 가 0.643913 으로 달라진다(§7 에 수치).""")

md(r"""## §1. 순전파 수식 (원문 3-1) — shape 표기

$$
\begin{aligned}
z_1 &= W_1 x + b_1 && W_1:(2,2),\ x:(2,),\ b_1:(2,) \ \Rightarrow\ z_1:(2,)\\
a_1 &= \sigma(z_1) = \frac{1}{1+e^{-z_1}} \ (\text{원소별}) && a_1:(2,)\\
z_2 &= W_2 \cdot a_1 + b_2 = \textstyle\sum_j W_{2,j}\, a_{1,j} + b_2 && W_2:(2,),\ b_2:() \ \Rightarrow\ z_2:()\ \text{스칼라}\\
\hat y &= y_{pred} = \sigma(z_2) && \hat y:()\\
L &= -\big(y \log \hat y + (1-y)\log(1-\hat y)\big) && L:()
\end{aligned}
$$

연결 그래프: $x \xrightarrow{W_1,b_1} z_1 \xrightarrow{\sigma} a_1 \xrightarrow{W_2,b_2} z_2 \xrightarrow{\sigma} \hat y \xrightarrow{\text{BCE}} L$""")

md(r"""## §2. 역전파 유도 (원문 3-2 · 3-5) — 출력층에서 입력층 방향으로

**연쇄 법칙이 필요한 이유**: $L$ 은 $W_1$ 을 직접 포함하지 않는다. $W_1$ 은 $z_1 \to a_1 \to z_2 \to \hat y \to L$ 을 거쳐서만
$L$ 에 영향을 주므로, 각 연결의 국소 미분을 곱해야 $\partial L/\partial W_1$ 이 나온다. 뒤에서 계산한 값($\delta$)을
앞 단계가 재사용하므로 계산량이 층 수에 선형이다 — 이것이 '역'전파의 이유다.

**보조 정리 — 시그모이드 미분**: $\sigma(z) = (1+e^{-z})^{-1}$ 이면
$\sigma'(z) = \dfrac{e^{-z}}{(1+e^{-z})^2} = \dfrac{1}{1+e^{-z}}\cdot\dfrac{e^{-z}}{1+e^{-z}} = \sigma(z)\,(1-\sigma(z))$.

**① ∂L/∂y_pred** — BCE 를 $\hat y$ 로 직접 미분 (shape ())

$$\frac{\partial L}{\partial \hat y} = -\frac{y}{\hat y} + \frac{1-y}{1-\hat y}$$

**② ∂L/∂z₂** — ①에 $\partial\hat y/\partial z_2 = \hat y(1-\hat y)$ 를 곱한다 (shape ())

$$\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial \hat y}\cdot\hat y(1-\hat y)
= \Big(-\frac{y}{\hat y} + \frac{1-y}{1-\hat y}\Big)\hat y(1-\hat y) = -y(1-\hat y) + (1-y)\hat y = \hat y - y \;\equiv\; \delta_2$$

(시그모이드 + BCE 조합에서 분모가 상쇄되어 $\hat y - y$ 가 된다. 이것은 결과이지 정의가 아니다.)

**③ ∂L/∂W₂** — $z_2 = \sum_j W_{2,j} a_{1,j} + b_2$ 이므로 $\partial z_2/\partial W_{2,j} = a_{1,j}$ (shape (2,))

$$\frac{\partial L}{\partial W_2} = \delta_2\, a_1, \qquad \frac{\partial L}{\partial b_2} = \delta_2$$

**④ ∂L/∂a₁** — $\partial z_2/\partial a_{1,j} = W_{2,j}$ (shape (2,))

$$\frac{\partial L}{\partial a_1} = \delta_2\, W_2$$

**⑤ ∂L/∂z₁** — 원소별 시그모이드이므로 야코비안이 대각 → 원소별 곱 ⊙ (shape (2,))

$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \odot a_1 \odot (1-a_1) \;\equiv\; \delta_1$$

**⑥ ∂L/∂W₁** — $z_{1,i} = \sum_j W_{1,ij} x_j + b_{1,i}$ 이므로 $\partial z_{1,i}/\partial W_{1,ij} = x_j$ (shape (2,2) = (2,)⊗(2,))

$$\frac{\partial L}{\partial W_{1,ij}} = \delta_{1,i}\, x_j \;\Longleftrightarrow\; \frac{\partial L}{\partial W_1} = \delta_1 x^{\top}\ (\text{외적}),
\qquad \frac{\partial L}{\partial b_1} = \delta_1$$

한 줄로 펼치면: $\dfrac{\partial L}{\partial W_{1,ij}} = \underbrace{\tfrac{\partial L}{\partial \hat y}}_{①}\,
\underbrace{\hat y(1-\hat y)}_{②}\, \underbrace{W_{2,i}}_{④}\, \underbrace{a_{1,i}(1-a_{1,i})}_{⑤}\, \underbrace{x_j}_{⑥}$

| # | 항 | 식 | shape |
|---|---|---|---|
| ① | ∂L/∂y_pred | −y/ŷ + (1−y)/(1−ŷ) | () |
| ② | ∂L/∂z₂ | ① · ŷ(1−ŷ) = ŷ − y | () |
| ③ | ∂L/∂W₂ | δ₂ · a₁ | (2,) |
| ④ | ∂L/∂a₁ | δ₂ · W₂ | (2,) |
| ⑤ | ∂L/∂z₁ | ④ ⊙ a₁(1−a₁) | (2,) |
| ⑥ | ∂L/∂W₁ | δ₁ xᵀ (외적) | (2,2) |
| + | ∂L/∂b₂, ∂L/∂b₁ | δ₂, δ₁ | (), (2,) |""")

md(r"""## §3. 순전파 손계산 (원문 3-3 · 3-4) — y_true = 1

입력 $x=[1,0]$, $W_1=\begin{bmatrix}0.1&0.2\\0.3&0.4\end{bmatrix}$, $b_1=[0,0]$, $W_2=[0.5,0.6]$, $b_2=0$.
계산기 방식: **매 단계 소수 6자리로 반올림**한 값을 다음 단계에 넣는다.

| 체크포인트 | 계산 | 값 | shape |
|---|---|---|---|
| $z_1$ | $[0.1\cdot1 + 0.2\cdot0,\ 0.3\cdot1 + 0.4\cdot0] + [0,0]$ | **[0.1, 0.3]** | (2,) |
| $e^{-z_1}$ | $[e^{-0.1},\ e^{-0.3}]$ | [0.904837, 0.740818] | (2,) |
| $a_1$ | $[1/1.904837,\ 1/1.740818]$ | **[0.524979, 0.574443]** | (2,) |
| $z_2$ | $0.5\cdot0.524979 + 0.6\cdot0.574443 + 0 = 0.2624895 + 0.3446658$ (곱은 반올림 전 7자리로 더한 뒤 6자리로) | **0.607155** | () |
| $e^{-z_2}$ | $e^{-0.607155}$ | 0.544899 | () |
| $\hat y$ (y_pred) | $1/1.544899$ | **0.647292** | () |
| $L$ | $-\ln 0.647292$ | **0.434958** | () |

(x 의 둘째 성분이 0 이라 $W_1$ 의 둘째 열(0.2, 0.4)은 $z_1$ 에 기여하지 않는다 → §4 에서 그 열의 기울기가 0.)""")

md(r"""## §4. 역전파 손계산 (원문 3-5 · 3-6) — y_true = 1, 각 단계 shape

| # | 항 | 계산 | 값 | shape |
|---|---|---|---|---|
| ① | ∂L/∂y_pred | $-1/\hat y = -1/0.647292$ (y=1 이라 둘째 항 0) | **−1.544898** | () |
| | $\hat y(1-\hat y)$ | $0.647292 \times 0.352708$ | 0.228305 | () |
| ② | ∂L/∂z₂ | $-1.544898 \times 0.228305$ | **−0.352708** | () |
| | 검산 | $\hat y - y = 0.647292 - 1$ | −0.352708 ✓ | () |
| ③ | ∂L/∂W₂ | $-0.352708 \times [0.524979,\ 0.574443]$ | **[−0.185164, −0.202611]** | (2,) |
| ④ | ∂L/∂a₁ | $-0.352708 \times [0.5,\ 0.6]$ | **[−0.176354, −0.211625]** | (2,) |
| | $a_1(1-a_1)$ | $[0.524979\times0.475021,\ 0.574443\times0.425557]$ | [0.249376, 0.244458] | (2,) |
| ⑤ | ∂L/∂z₁ | $[-0.176354\times0.249376,\ -0.211625\times0.244458]$ | **[−0.043978, −0.051733]** | (2,) |
| ⑥ | ∂L/∂W₁ | $\delta_1 x^\top = \begin{bmatrix}-0.043978\cdot1 & -0.043978\cdot0\\ -0.051733\cdot1 & -0.051733\cdot0\end{bmatrix}$ | **[[−0.043978, 0], [−0.051733, 0]]** | (2,2) |
| + | ∂L/∂b₂ · ∂L/∂b₁ | $\delta_2$ · $\delta_1$ | −0.352708 · [−0.043978, −0.051733] | () · (2,) |

참고: ① 은 $-1/\hat y = -(1+e^{-z_2}) = -1.544899$ 로도 쓸 수 있다. 6자리 반올림 경로에 따라 마지막 자리가
−1.544898 / −1.544899 로 갈리는데, 원문 기준(소수 4자리)에서는 둘 다 −1.5449 다.""")

code(r"""# §5. NumPy 로 같은 계산 (원문 3-7) — src 모듈을 쓰지 않는 노트북 안의 독립 구현
import numpy as np

x  = np.array([1.0, 0.0])                    # (2,)
W1 = np.array([[0.1, 0.2], [0.3, 0.4]])      # (2,2)  행 = 은닉 뉴런
b1 = np.zeros(2)                             # (2,)
W2 = np.array([0.5, 0.6])                    # (2,)
b2 = 0.0                                     # ()
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))

def forward_backward(y):
    z1 = W1 @ x + b1;  a1 = sigmoid(z1)                       # (2,), (2,)
    z2 = W2 @ a1 + b2; yhat = sigmoid(z2)                     # (), ()
    L = -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))      # ()
    dyhat = -y / yhat + (1 - y) / (1 - yhat)                  # ① ()
    dz2 = dyhat * yhat * (1 - yhat)                           # ② ()
    dW2 = dz2 * a1                                            # ③ (2,)
    da1 = dz2 * W2                                            # ④ (2,)
    dz1 = da1 * a1 * (1 - a1)                                 # ⑤ (2,)
    dW1 = np.outer(dz1, x)                                    # ⑥ (2,2)
    return dict(z1=z1, a1=a1, z2=z2, y_pred=yhat, L=L, dL_dy_pred=dyhat, dL_dz2=dz2,
                dL_dW2=dW2, dL_da1=da1, dL_dz1=dz1, dL_dW1=dW1, dL_db2=dz2, dL_db1=dz1)

np_vals = forward_backward(1.0)
for k, v in np_vals.items():
    print(f"{k:11s} shape={str(np.shape(v)):6s} {np.round(v, 6)}")""")

code(r"""# 손계산 값(§3·§4 표에서 그대로 옮겨 적은 상수) vs NumPy — 소수 4자리 일치 확인
HAND_Y1 = {
    "z1": [0.1, 0.3], "a1": [0.524979, 0.574443], "z2": 0.607155, "y_pred": 0.647292, "L": 0.434958,
    "dL_dy_pred": -1.544898, "dL_dz2": -0.352708, "dL_dW2": [-0.185164, -0.202611],
    "dL_da1": [-0.176354, -0.211625], "dL_dz1": [-0.043978, -0.051733],
    "dL_dW1": [[-0.043978, 0.0], [-0.051733, 0.0]], "dL_db2": -0.352708, "dL_db1": [-0.043978, -0.051733],
}

def compare(hand, num, label):
    ok_all = True
    print(f"[{label}] {'항':11s} {'손(4자리)':>22s} {'NumPy(4자리)':>22s}  최대|차|    4자리일치")
    for k, hv in hand.items():
        h, n = np.asarray(hv, dtype=float), np.asarray(num[k], dtype=float)
        assert h.shape == n.shape, (k, h.shape, n.shape)          # shape 도 같아야 한다
        diff = float(np.max(np.abs(h - n)))
        ok = bool(np.array_equal(np.round(h, 4), np.round(n, 4)) and diff < 5e-5)
        ok_all &= ok
        print(f"[{label}] {k:11s} {str(np.round(h, 4).tolist()):>22s} {str(np.round(n, 4).tolist()):>22s}  {diff:.1e}   {'✓' if ok else '✗'}")
    return ok_all

ok1 = compare(HAND_Y1, np_vals, "y=1")
print("\n원문 3-7 (y_true=1) 손계산 = NumPy (소수 4자리):", ok1)
assert ok1""")

code(r"""# §6. gradcheck — 해석식 자체가 맞는지 손실 함수를 직접 흔들어(중심차분) 확인한다 (해석식과 독립)
import pathlib, sys
ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / 'src' / 'probability.py').is_file())
sys.path.insert(0, str(ROOT / 'src'))
import backprop as B          # 제출 모듈(src/backprop.py) 도 같은 값을 내는지 함께 확인

for y in (1.0, 0.0):
    h, n = B.backward(y), B.numeric_grad(y)
    for k in ('dW1', 'dW2', 'db1', 'db2'):
        e = B.grad_errs(h[k], n[k])
        print(f"y={y} gradcheck {k}: 상대오차(0 제외) {e['상대오차(0이_아닌_항)']:.3e} · 절대오차 {e['절대오차(전체)']:.3e}")
    nb = forward_backward(y)
    same = all(np.allclose(h[a], nb[b], atol=1e-12) for a, b in
               (('dW1', 'dL_dW1'), ('dW2', 'dL_dW2'), ('dz2', 'dL_dz2'), ('da1', 'dL_da1'), ('dy_pred', 'dL_dy_pred')))
    print(f"y={y} src/backprop.py == 노트북 NumPy 구현: {same}")""")

md(r"""## §7. 원문 공란·관례에 대한 대조

### y_true = 0 일 때 (같은 순전파, ŷ = 0.647292)

| # | 항 | 계산 | 값(손, 6자리) |
|---|---|---|---|
| | L | $-\ln(1-0.647292) = -\ln 0.352708$ | 1.042115 |
| ① | ∂L/∂y_pred | $1/(1-\hat y) = 1/0.352708$ | 2.835206 |
| ② | ∂L/∂z₂ | $2.835206 \times 0.228305$ (= ŷ − 0) | 0.647292 |
| ③ | ∂L/∂W₂ | $0.647292 \times a_1$ | [0.339815, 0.371832] |
| ④ | ∂L/∂a₁ | $0.647292 \times W_2$ | [0.323646, 0.388375] |
| ⑤ | ∂L/∂z₁ | ④ ⊙ [0.249376, 0.244458] | [0.080710, 0.094941] |
| ⑥ | ∂L/∂W₁ | $\delta_1 x^\top$ | [[0.080710, 0], [0.094941, 0]] |

y 를 바꾸면 순전파(z1·a1·z2·ŷ)는 그대로이고 L 과 모든 기울기의 **부호와 크기**가 바뀐다(단순 부호 반전이 아니다:
δ₂ 가 −0.352708 → +0.647292).""")

code(r"""HAND_Y0 = {"L": 1.042115, "dL_dy_pred": 2.835206, "dL_dz2": 0.647292, "dL_dW2": [0.339815, 0.371832],
           "dL_da1": [0.323646, 0.388375], "dL_dz1": [0.080710, 0.094941], "dL_dW1": [[0.080710, 0.0], [0.094941, 0.0]]}
ok0 = compare(HAND_Y0, forward_backward(0.0), "y=0")
print("\ny_true=0 손계산 = NumPy (소수 4자리):", ok0)

# 관례 대조: z1 = x @ W1 (행벡터 관례)로 읽으면 숫자가 달라진다
z1_alt = x @ W1; a1_alt = sigmoid(z1_alt); yhat_alt = sigmoid(W2 @ a1_alt + b2)
print("\n관례 대조  z1 = W1 @ x :", (W1 @ x).tolist(), " → y_pred", round(float(np_vals['y_pred']), 6))
print("관례 대조  z1 = x @ W1 :", z1_alt.tolist(), " → y_pred", round(float(yhat_alt), 6), "(이 노트북은 W1 @ x 를 쓴다)")""")

md(r"""## §8. 정리

- ∂L/∂W₁ 의 둘째 열이 0 인 것은 오류가 아니라 **입력 결핍**($x_2 = 0$)이다. 그래서 gradcheck 는 '0 이 아닌 항이 존재'하는지와
  0 항의 절대오차를 따로 본다(상대오차는 분모 0 에서 폭주).
- ② 에서 $\hat y - y$ 가 나오는 상쇄는 시그모이드+BCE 짝의 성질이다. MSE+시그모이드였다면 $(\hat y - y)\hat y(1-\hat y)$ 가 되어
  ŷ 가 0·1 근처에서 기울기가 사라진다 — 분류에 BCE 를 쓰는 실무적 이유이자, 확률 노트북의 'CE = 베르누이 MLE' 와 이어진다.""")
