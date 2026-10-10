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


def _from_cells(cells) -> "nbf.NotebookNode":
    nb = nbf.v4.new_notebook()
    for kind, text in cells:
        (md if kind == "md" else code)(nb, text)
    return nb


def backprop_nb():
    """셀 정의는 scripts/_nb_backprop.py (2026-10-11: 유도·손계산·독립 NumPy 대조를 분리된 절로 재작성)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import _nb_backprop
    return _from_cells(_nb_backprop.CELLS)


def probability_nb():
    """셀 정의는 scripts/_nb_probability.py (2026-10-11: MSE·CE=MLE LaTeX 유도를 노트북 안에, 카테고리 분포 추가)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import _nb_probability
    return _from_cells(_nb_probability.CELLS)


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
    # 커널의 파이썬 패치 버전(3.13.14 → 3.13.16)만 달라도 diff 가 났다(2026-10-11 재실행에서 관측) → 버전 문자열 제거
    nb.metadata.get("language_info", {}).pop("version", None)


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
