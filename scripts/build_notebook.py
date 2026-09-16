"""notebooks/ 2종을 만들고 **실행해서** 결과를 박아 저장한다(nbconvert 대신 nbclient 내장).

미션 요구: shape 표기·손계산 과정은 마크다운으로 (코드 주석 금지). 그래서 md()/code() 를 번갈아 넣고,
실행 후 각 셀의 출력을 그대로 보관한다 → 클론 직후에도 '실행된 노트북'으로 제출된다.
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
            "규칙을 정하지 않으면 shape 이 꼬인다. 이 노트의 규칙:\n\n"
            "| 수식 | 코드 | shape |\n|---|---|---|\n"
            "| z₁ = W₁x + b₁ | `z1 = W1 @ x + b1` | (2,) |\n"
            "| a₁ = σ(z₁) | `a1 = 1/(1+exp(-z1))` | (2,) |\n"
            "| z₂ = W₂ᵗa₁ + b₂ | `z2 = W2 @ a1 + b2` | () |\n"
            "| ŷ = σ(z₂) | `yhat = 1/(1+exp(-z2))` | () |\n"
            "| L = −[y log ŷ + (1−y) log(1−ŷ)] | `bce(yhat, y)` | () |\n"
            "| ∂L/∂z₂ = ŷ − y | `dz2 = yhat - y` | () |\n"
            "| ∂L/∂W₂ = a₁·dz₂ | `dW2 = a1 * dz2` | (2,) |\n"
            "| ∂L/∂z₁ = (W₂ dz₂) ⊙ a₁(1−a₁) | `dz1 = (W2*dz2) * a1*(1-a1)` | (2,) |\n"
            "| ∂L/∂W₁ = dz₁ xᵗ | `dW1 = np.outer(dz1, x)` | (2,2) |\n")
    code(nb, "import numpy as np, pathlib, sys\n"
             "sys.path.insert(0, str(pathlib.Path.cwd() / 'src'))\n"
             "import backprop as B\n"
             "x, W1, b1, W2, b2 = B.X, B.W1, B.B1, B.W2, B.B2\n"
             "print('W1', W1.shape, 'x', x.shape, 'W2', W2.shape)")
    md(nb, "## 1. 순전파: 숫자를 하나씩")
    code(nb, "z1, a1, z2, yhat = B.forward()\n"
             "print('z1 =', z1, '\\na1 =', a1, '\\nz2 =', round(z2,6), '\\nŷ =', round(yhat,6))")
    md(nb, "## 2. 손으로 확인 (σ(0.1)=0.524979 …)\n\n"
            "z₁ = W₁x+b₁ = [0.1, 0.3]\na₁ = σ(z₁) = [0.524979, 0.574443]\n"
            "z₂ = 0.5·0.524979 + 0.6·0.574443 = 0.607155\nŷ = σ(0.607155) = 0.647292\n"
            "L(y=1) = −log 0.647292 = 0.434958")
    code(nb, "print('손계산 대조:', abs(round(float(yhat),6) - 0.647292) < 1e-6, abs(round(float(B.bce(yhat,1.0)),6) - 0.434958) < 1e-6)")
    md(nb, "## 3. 역전파 + 수치미분 검산\n\n"
            "검산은 **같은 수식을 뒤로 돌리는** 방식이 아니라 손실 함수를 흔드는 방식이어야 항등식이 되지 않는다.")
    cell = '\n'.join([
        "for y in (1.0, 0.0):",
        "    h, n = B.backward(y), B.numeric_grad(y)",
        "    e = B.grad_errs(h['dW1'], n['dW1'])",
        "    print('y=', y, 'L=', round(h['L'], 6), 'dz2=', round(float(h['dz2']), 6))",
        "    print('   dW1 해석', h['dW1'].ravel(), '  / 수치', n['dW1'].ravel())",
        "    print('   상대오차(0 제외)', e['상대오차(0이_아닌_항)'], ' 절대오차', e['절대오차(전체)'])",
    ])
    code(nb, cell)
    md(nb, "## 4. 왜 ∂L/∂W₁ 의 두 번째 열이 0인가\n\n"
            "x=[1,0] 이라 W₁ 의 두 번째 열은 순전파에 참여하지 않는다. **기울기 0 = 오동작이 아니라 입력 결핍.**\n"
            "그래서 검사는 '모든 항이 0이 아니어야 한다'가 아니라 '0이 아닌 항이 존재해야 한다'여야 한다.")
    code(nb, "dW1 = B.backward(1.0)['dW1']\nprint('0이 아닌 항 개수:', int(np.count_nonzero(dW1)), '/', dW1.size)")
    md(nb, "## 5. y_true 가 미션에 없다 (결함)\n\n"
            "정의서가 비어 있어 **양쪽**을 다 실었다. 리포트는 y=1.0 을 표준으로 쓰되, 이 표를 함께 인용한다.")
    code(nb, "import json, pathlib\nd = json.loads(pathlib.Path('reports/backprop.json').read_text(encoding='utf-8'))\n"
             "print(json.dumps({k: {kk: v[kk] for kk in ('L','dz2','dW2','dW1','최대상대오차_0제외')} for k, v in d['y_true_별'].items()},\n"
             "                 ensure_ascii=False, indent=1))")
    return nb


def probability_nb():
    nb = nbf.v4.new_notebook()
    md(nb, "# 손실 함수는 선택이 아니라 결론이다\n\n"
            "1. y = f(x) + ε, ε~N(0,σ²) 라면 **MSE** 는 로그우도를 최대화하는 것과 같다(σ 고정).\n"
            "2. y∈{0,1} ~ Bernoulli(p) 라면 **교차엔트로피**를 최소화하는 p̂ 는 우도를 최대화하는 p̂ 와 같다.\n"
            "여기서 격자해상도 0.001 이 '완전 일치'의 뜻을  규정한다. 격자를 굵게 쓰면 0이 아니게 된다.")
    code(nb, "import numpy as np, json, pathlib, sys\nsys.path.insert(0, str(pathlib.Path.cwd() / 'src'))\n"
             "import probability as P\nd = json.loads(pathlib.Path('reports/probability.json').read_text(encoding='utf-8'))\n"
             "print(json.dumps(d['MSE_MLE_동치'], ensure_ascii=False, indent=1))")
    md(nb, "## 정규·베르누이 분포 그림\n\n그림 파일은 `figures/06_normal_pdf.png`, `figures/07_bernoulli_pmf.png`. "
            "넓이가 1이 되는 건 **적분 구간이 무한할 때**이고, [-8,8] 에서 σ=3 은 0.9923 으로 잘린다.")
    code(nb, "x = np.linspace(-8, 8, 400)\nfor s in (1.0, 3.0):\n"
             "    print(f'σ={s}: peak={P.normal_pdf(0,0,s):.6f} ∫= {np.trapezoid(P.normal_pdf(x,0,s), x):.6f}')")
    md(nb, "## 교차엔트로피 = MLE, 그리고 깁스 부등식의 범위")
    code(nb, "print(json.dumps(d['CE_MLE_동치'], ensure_ascii=False, indent=1))")
    md(nb, "플러그인(표본 자체) 엔트로피는 참값 엔트로피보다 **작게** 나온다(아래 편향). "
           "그러므로 '손실이 엔트로피보다 크면 이상하다'는 문장은 표본 기준으로는 틀린다.")
    code(nb, "c = d['CE_MLE_동치']\nprint('플러그인', c['플러그인_엔트로피'], '참값 H', c['진_엔트로피'], '편향', c['플러그인_편향'],\n      '참값 대비 CE', c['참값대_ fitted_교차엔트로피'], '깁스 성립', c['깁스_성립'])")
    md(nb, "## softmax 의 수치 안정성은 '큰 스케일에서만' 검증된다")
    code(nb, "z = np.array([1.,2.,3.])\nwith np.errstate(over='ignore'):\n"
             "    print('원본 차이', np.max(np.abs(P.softmax(z,False)-P.softmax(z,True))))\n"
             "    print('×1000 naive', P.softmax(z*1000,False), '\\n×1000 safe  ', P.softmax(z*1000,True))")
    return nb


def main() -> None:
    from nbclient import NotebookClient
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, builder in (("backprop_derivation.ipynb", backprop_nb), ("probability_loss.ipynb", probability_nb)):
        nb = builder()
        # 커널 cwd 를 레포 루트로 둔다(그래야 src·reports 가 한 경로로 잡힌다)
        NotebookClient(nb, timeout=300, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
        nbf.write(nb, out / name)
        exec_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
        print(f"{name}: 셀 {len(nb.cells)}개(출력 {exec_out}개) → {out / name}")


if __name__ == "__main__":
    main()
