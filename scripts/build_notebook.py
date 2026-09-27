"""notebooks/ 2종을 만들고 **실행해서** 결과를 박아 저장한다(nbconvert 대신 nbclient 내장).

미션 요구: shape 표기·손계산 과정은 마크다운으로 (코드 주석 금지). 그래서 md()/code() 를 번갈아 넣고,
실행 후 각 셀의 출력을 그대로 보관한다 → 클론 직후에도 '실행된 노트북'으로 제출된다.

감사(2026-09-27) 반영:
- 역전파 노트북: 원문 3-5 필수 유도 **6항 전체**(∂L/∂y_pred, ∂L/∂z2, ∂L/∂W2, ∂L/∂a1, ∂L/∂z1, ∂L/∂W1)
  + 편향 ∂L/∂b1, ∂L/∂b2 를 룰표와 출력에 넣었다(이전엔 ∂L/∂z₂ = ŷ−y 로 직행해 두 항이 빠져 있었다).
- 확률 노트북: 원문 제출물 5 의 "확률 분포 시각화"를 **노트북 안에 그림으로** embed 한다
  (이전엔 텍스트 print 만 있었고 그림은 src 가 따로 만들었다). 분포는 원문 지정 N(0,1)·N(2,0.5) / B(0.3)·B(0.7).
  softmax 합=1(±1e-6) 검사도 노트북에서 직접 돌린다.
"""
from __future__ import annotations

import pathlib
import sys

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def md(nb, text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))


def code(nb, src):
    nb.cells.append(nbf.v4.new_code_cell(src))


def backprop_nb():
    nb = nbf.v4.new_notebook()
    md(nb, "# 역전파를 손으로 (2-2-1)\n\n"
            "규칙을 정하지 않으면 shape 이 꼬인다. 이 노트의 규칙(원문 3-5 필수 유도 6항 + 편향 2항):\n\n"
            "| # | 수식 | 코드 | shape |\n|---|---|---|---|\n"
            "| 순 | z₁ = W₁x + b₁ | `z1 = W1 @ x + b1` | (2,) |\n"
            "| 순 | a₁ = σ(z₁) | `a1 = 1/(1+exp(-z1))` | (2,) |\n"
            "| 순 | z₂ = W₂ᵗa₁ + b₂ | `z2 = W2 @ a1 + b2` | () |\n"
            "| 순 | ŷ = σ(z₂) | `yhat = 1/(1+exp(-z2))` | () |\n"
            "| 순 | L = −[y log ŷ + (1−y) log(1−ŷ)] | `bce(yhat, y)` | () |\n"
            "| **1** | ∂L/∂y_pred = −y/ŷ + (1−y)/(1−ŷ) | `dy_pred = -(y/yhat) + (1-y)/(1-yhat)` | () |\n"
            "| **2** | ∂L/∂z₂ = ∂L/∂y_pred · ŷ(1−ŷ) = ŷ − y | `dz2 = dy_pred * yhat*(1-yhat)` | () |\n"
            "| **3** | ∂L/∂W₂ = a₁ · ∂L/∂z₂ | `dW2 = a1 * dz2` | (2,) |\n"
            "| **4** | ∂L/∂a₁ = W₂ · ∂L/∂z₂ | `da1 = W2 * dz2` | (2,) |\n"
            "| **5** | ∂L/∂z₁ = ∂L/∂a₁ ⊙ a₁(1−a₁) | `dz1 = da1 * a1*(1-a1)` | (2,) |\n"
            "| **6** | ∂L/∂W₁ = ∂L/∂z₁ xᵗ | `dW1 = np.outer(dz1, x)` | (2,2) |\n"
            "| + | ∂L/∂b₂ = ∂L/∂z₂ | `db2 = dz2` | () |\n"
            "| + | ∂L/∂b₁ = ∂L/∂z₁ | `db1 = dz1` | (2,) |\n\n"
            "**1~6 이 원문이 요구한 필수 유도다.** 2번은 1번을 거쳐야 나온다(시그모이드와 BCE 가 상쇄해 ŷ−y 가 되는 것이지, "
            "처음부터 ŷ−y 가 정의가 아니다). 이전 버전 노트는 1·4번을 생략하고 2번부터 시작했다 — 감사에서 지적돼 복원했다.")
    code(nb, "import numpy as np, pathlib, sys\n"
             "sys.path.insert(0, str(pathlib.Path.cwd() / 'src'))\n"
             "import backprop as B\n"
             "x, W1, b1, W2, b2 = B.X, B.W1, B.B1, B.W2, B.B2\n"
             "print('W1', W1.shape, 'x', x.shape, 'b1', b1.shape, 'W2', W2.shape, 'b2', np.shape(b2))")
    md(nb, "## 1. 순전파: 숫자를 하나씩 (원문 3-4 중간값 z1, a1, z2, y_pred 기록)")
    code(nb, "z1, a1, z2, yhat = B.forward()\n"
             "print('z1 =', z1, '\\na1 =', a1, '\\nz2 =', round(z2,6), '\\ny_pred = ŷ =', round(yhat,6))")
    md(nb, "## 2. 손으로 확인 (σ(0.1)=0.524979 …)\n\n"
            "z₁ = W₁x+b₁ = [0.1, 0.3]\na₁ = σ(z₁) = [0.524979, 0.574443]\n"
            "z₂ = 0.5·0.524979 + 0.6·0.574443 = 0.607155\nŷ = σ(0.607155) = 0.647292\n"
            "L(y=1) = −log 0.647292 = 0.434958")
    code(nb, "print('손계산 대조:', abs(round(float(yhat),6) - 0.647292) < 1e-6, abs(round(float(B.bce(yhat,1.0)),6) - 0.434958) < 1e-6)")
    md(nb, "## 3. 역전파 6항 + 편향 — 손값과 코드를 나란히\n\n"
            "검산은 **같은 수식을 뒤로 돌리는** 방식이 아니라 손실 함수를 흔드는 방식이어야 항등식이 되지 않는다.")
    cell = '\n'.join([
        "for y in (1.0, 0.0):",
        "    h, n = B.backward(y), B.numeric_grad(y)",
        "    print('=== y_true =', y, '===')",
        "    print(' L           =', round(h['L'], 6))",
        "    print(' ∂L/∂y_pred  =', round(h['dy_pred'], 6), '  손: -(y/ŷ)+(1-y)/(1-ŷ) =',"
        " round(-(y/0.647292)+(1-y)/(1-0.647292), 6))",
        "    print(' ∂L/∂z₂      =', round(float(h['dz2']), 6), ' = ∂L/∂y_pred·ŷ(1-ŷ) =',"
        " round(h['dy_pred']*0.647292*(1-0.647292), 6), ' = ŷ-y =', round(0.647292-y, 6))",
        "    print(' ∂L/∂W₂      =', np.round(h['dW2'], 6), ' shape', h['dW2'].shape)",
        "    print(' ∂L/∂a₁      =', np.round(h['da1'], 6), ' shape', h['da1'].shape, '  손: W₂·∂L/∂z₂ =',"
        " np.round(np.array([0.5,0.6])*float(h['dz2']), 6))",
        "    print(' ∂L/∂z₁      =', np.round(h['dz1'], 6), ' shape', h['dz1'].shape)",
        "    print(' ∂L/∂W₁      =', np.round(h['dW1'], 6), ' shape', h['dW1'].shape)",
        "    print(' ∂L/∂b₁      =', np.round(h['db1'], 6), ' / ∂L/∂b₂ =', round(float(h['db2']), 6))",
        "    for k in ('dW1','dW2','db1','db2'):",
        "        e = B.grad_errs(h[k], n[k])",
        "        print(f'   gradcheck {k}: 상대오차(0 제외) {e[\"상대오차(0이_아닌_항)\"]:.3e} · 절대오차 {e[\"절대오차(전체)\"]:.3e}')",
    ])
    code(nb, cell)
    md(nb, "## 4. 왜 ∂L/∂W₁ 의 두 번째 열이 0인가\n\n"
            "x=[1,0] 이라 W₁ 의 두 번째 열은 순전파에 참여하지 않는다. **기울기 0 = 오동작이 아니라 입력 결핍.**\n"
            "그래서 검사는 '모든 항이 0이 아니어야 한다'가 아니라 '0이 아닌 항이 존재해야 한다'여야 한다.\n"
            "같은 이유로 ∂L/∂a₁ 의 두 성분은 둘 다 살아 있다(W₂=[0.5,0.6] 이 0이 아니므로).")
    code(nb, "dW1 = B.backward(1.0)['dW1']\nprint('0이 아닌 항 개수:', int(np.count_nonzero(dW1)), '/', dW1.size)\n"
             "print('da1 0이 아닌 항:', int(np.count_nonzero(B.backward(1.0)['da1'])), '/', B.backward(1.0)['da1'].size)")
    md(nb, "## 5. y_true 가 미션에 없다 (원문 결함)\n\n"
            "정의서가 비어 있어 **양쪽**을 다 실었다. 리포트는 y=1.0 을 표준으로 쓰되, 이 표를 함께 인용한다.")
    code(nb, "import json, pathlib\n"
             "d = json.loads(pathlib.Path('reports/backprop.json').read_text(encoding='utf-8'))\n"
             "print('원문 3-5 필수유도 표:'); print(json.dumps(d['원문3-5_필수유도']['표'], ensure_ascii=False, indent=1))\n"
             "print('y_true 별 6항 값:'); print(json.dumps(d['원문3-5_필수유도']['y_true_별_값'], ensure_ascii=False, indent=1))\n"
             "print('게이트:', json.dumps({k: {kk: v[kk] for kk in ('L','dz2','dW2','dW1','최대상대오차_0제외')} for k, v in d['y_true_별'].items()},\n"
             "                 ensure_ascii=False, indent=1))")
    return nb


def probability_nb():
    nb = nbf.v4.new_notebook()
    md(nb, "# 손실 함수는 선택이 아니라 결론이다\n\n"
            "1. y = f(x) + ε, ε~N(0,σ²) 라면 **MSE** 는 로그우도를 최대화하는 것과 같다(σ 고정).\n"
            "2. y∈{0,1} ~ Bernoulli(p) 라면 **교차엔트로피**를 최소화하는 p̂ 는 우도를 최대화하는 p̂ 와 같다.\n"
            "여기서 격자해상도 0.001 이 '완전 일치'의 뜻을 규정한다(같은 격자 위에서는 두 목적함수가 단조관계라 "
            "argmin 이 갈리지 않는다 — 격자를 바꾼 '갈린다' 주장은 실측 없이 쓰지 않는다).")
    # 분포 그림은 plt.show() 에 맡기지 않고 **display_data(image/png) 로 명시적으로 embed** 한다.
    # 이유: src/_common 이 headless 에서 Agg 를 켜므로 백엔드 상태에 따라 plt.show() 출력이 사라질 수 있다.
    # (감사 반영 — 원문 제출물 5 는 '노트북의 확률 분포 시각화'를 요구한다)
    code(nb, "import numpy as np, json, pathlib, sys\n"
             "import matplotlib, matplotlib.pyplot as plt\n"
             "from IPython import display as _ipd\n"
             "sys.path.insert(0, str(pathlib.Path.cwd() / 'src'))\n"
             "import probability as P\n"
             "import _common  # fig_to_png_b64 (그림 → base64 PNG)\n"
             "d = json.loads(pathlib.Path('reports/probability.json').read_text(encoding='utf-8'))\n"
             "print(json.dumps(d['MSE_MLE_동치'], ensure_ascii=False, indent=1))")
    md(nb, "## 확률 분포 시각화 (원문 제출물 5 — 이 노트북 안에 그림으로)\n\n"
            "원문 5-1: 정규분포 **N(0,1)** 과 **N(2,0.5)** 의 PDF 를 하나의 Figure에.\n"
            "원문 5-2: 베르누이 **B(0.3)** 과 **B(0.7)** 의 PMF 막대그래프를 하나의 Figure에.\n\n"
            "베르누이는 **시행 1회**라 k∈{0,1} 두 막대이고 합이 1 이어야 한다. "
            "(이전 구현은 `p^k(1-p)^(10-k)`, k=0..10 을 '베르누이'라 라벨링했는데 합이 0.647/0.249 로 1 이 아니었다 "
            "→ 이항계수 C(10,k) 가 없는 거듭제곱이라 확률분포가 아니다. 감사에서 정정했고, "
            "당초 배우려던 Binomial(10,p) 는 아래에 따로 그린다.)")
    code(nb, "x = np.linspace(-8, 8, 1601)\n"
             "fig, ax = plt.subplots(figsize=(6.0, 3.2))\n"
             "for label, (mu, s) in P.NORMALS.items():\n"
             "    ax.plot(x, P.normal_pdf(x, mu, s), lw=1.5, label=f'{label} peak={P.normal_pdf(mu, mu, s):.4f}')\n"
             "ax.set_title('원문 5-1: N(0,1) vs N(2,0.5) PDF'); ax.set_xlabel('x'); ax.set_ylabel('밀도'); ax.legend(fontsize=7)\n"
             "_ipd.display(_ipd.Image(data=P.fig_to_png_bytes(fig), format='png'))\n"
             "for label, (mu, s) in P.NORMALS.items():\n"
             "    print(f'{label}: peak={P.normal_pdf(mu,mu,s):.6f}  ∫[-8,8]={np.trapezoid(P.normal_pdf(x,mu,s),x):.6f}')")
    code(nb, "k = np.array([0, 1])\n"
             "fig, ax = plt.subplots(figsize=(6.0, 3.2))\n"
             "for i, (label, p) in enumerate(P.BERNOULLIS.items()):\n"
             "    pmf = P.bernoulli_pmf(p)\n"
             "    ax.bar(k + (-0.18 if i == 0 else 0.18), pmf, width=.34, label=f'{label} 합={pmf.sum():.6f}')\n"
             "ax.set_xticks([0,1]); ax.set_ylim(0,.85); ax.set_title('원문 5-2: 베르누이 PMF B(0.3) vs B(0.7)')\n"
             "ax.set_xlabel('성공 횟수 k'); ax.set_ylabel('P(X=k)'); ax.legend(fontsize=7)\n"
             "_ipd.display(_ipd.Image(data=P.fig_to_png_bytes(fig), format='png'))\n"
             "print(json.dumps(d['베르누이PMF(원문5-2)'], ensure_ascii=False, indent=1))\n"
             "print('이전 구현(잘못된 라벨)의 막대 합:', json.dumps(d['베르누이PMF_정정_기록']['이전_구현_막대합'], ensure_ascii=False))")
    code(nb, "kk = np.arange(0, 11)\n"
             "fig, ax = plt.subplots(figsize=(6.0, 3.0))\n"
             "for p, off in ((0.3, -0.2), (0.7, 0.2)):\n"
             "    ax.bar(kk + off, P.binomial_pmf(10, p), width=.38, label=f'Binomial(10,{p}) 합={P.binomial_pmf(10,p).sum():.6f}')\n"
             "ax.set_title('대조: 이항분포 Binomial(10,p) — 이항계수를 넣어야 합이 1'); ax.set_xlabel('성공 횟수 k')\n"
             "ax.set_ylabel('P(X=k)'); ax.legend(fontsize=7)\n"
             "_ipd.display(_ipd.Image(data=P.fig_to_png_bytes(fig), format='png'))")
    md(nb, "## 교차엔트로피 = MLE, 그리고 깁스 부등식의 범위")
    code(nb, "print(json.dumps(d['CE_MLE_동치'], ensure_ascii=False, indent=1))")
    md(nb, "플러그인(표본 자체) 엔트로피는 참값 엔트로피보다 **작게** 나온다(아래 편향). "
           "그러므로 '손실이 엔트로피보다 크면 이상하다'는 문장은 표본 기준으로는 틀린다.")
    code(nb, "c = d['CE_MLE_동치']\nprint('플러그인', c['플러그인_엔트로피'], '참값 H', c['진_엔트로피'], '편향', c['플러그인_편향'],\n      '참값 대비 CE', c['참값대_ fitted_교차엔트로피'], '깁스 성립', c['깁스_성립'])")
    md(nb, "## softmax: 원문 5-3 (출력 합 = 1.0 ± 1e-6) 과 수치 안정성\n\n"
            "안정화(max 차감)의 효과는 **큰 스케일에서만** 관측된다. 작은 스케일에서의 차이는 1e-16 수준이라 검사로서 무의미하다.")
    code(nb, "z = np.array([1.,2.,3.])\n"
             "print(json.dumps(d['softmax']['합=1_검증(원문5-3)'], ensure_ascii=False, indent=1))\n"
             "print('gate 합=1 ±1e-6:', d['softmax']['gate_합=1_±1e-6'], '· 최대 오차', d['softmax']['최대_합-1_절대오차'])\n"
             "with np.errstate(over='ignore', invalid='ignore'):\n"
             "    print('원본 차이', np.max(np.abs(P.softmax(z,False)-P.softmax(z,True))))\n"
             "    print('×1000 naive', P.softmax(z*1000,False), '\\n×1000 safe  ', P.softmax(z*1000,True))")
    return nb


def stabilize(nb) -> None:
    """재빌드해도 바이트가 같도록 두 가지를 고정한다(재현성 주장의 최소 조건).

    1. 셀 id — nbformat 기본값은 무작위라 재빌드마다 diff 가 난다 → (순서+내용) 해시로.
    2. 출력 metadata 의 실행 시각(iopub.execute_input 등) — 커밋된 노트북에 매번 새 타임스탬프가 박힌다 → 제거.
    (PNG 바이트와 execution_count 는 결정론적이라 그대로 둔다.)
    """
    import hashlib
    for i, c in enumerate(nb.cells):
        src = "".join(c.get("source", [])) if isinstance(c.get("source"), list) else c.get("source", "")
        c["id"] = hashlib.sha1(f"{i}:{c['cell_type']}:{src}".encode("utf-8")).hexdigest()[:12]
        c.get("metadata", {}).pop("execution", None)   # 셀 단위 실행 시각(타임스탬프) 제거
        for o in c.get("outputs", []):
            o["metadata"] = {}


def main() -> None:
    from nbclient import NotebookClient
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, builder in (("backprop_derivation.ipynb", backprop_nb), ("probability_loss.ipynb", probability_nb)):
        nb = builder()
        # 커널 cwd 를 레포 루트로 둔다(그래야 src·reports 가 한 경로로 잡힌다)
        NotebookClient(nb, timeout=300, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        stabilize(nb)
        nbf.write(nb, out / name)
        exec_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
        imgs = sum(1 for c in nb.cells if c.cell_type == "code"
                   for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
        print(f"{name}: 셀 {len(nb.cells)}개(출력 {exec_out}개, 그림 {imgs}장) → {out / name}")


if __name__ == "__main__":
    main()
