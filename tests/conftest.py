"""게이트 테스트의 실행 전략.

- 기본: 구현이 없으면 **skip**(레포가 초록으로 보이지만 README가 '몇 건이 skip인지'를 말한다).
- `MISSION_STRICT=1 pytest`: skip → fail. 제출 직전에는 반드시 이 모드(미구현을 남기지 않기 위한 장치).
- '통과하지만 무의미한 테스트'를 피하려고, 모든 게이트는 **측정 대상 파일**이 있을 때만 실행된다.
"""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
STRICT = os.environ.get("MISSION_STRICT") == "1"


def implemented() -> bool:
    """미션 코드가 한 줄이라도 있는가. 없으면 게이트 검사는 판정할 대상이 없다."""
    return (ROOT / "reports" / "gates.json").exists() or any((ROOT / d).exists() for d in ("src", "figures", "notebooks"))


BOOTSTRAP = os.environ.get("MISSION_BOOTSTRAP") == "1"


def ensure_inputs():
    """파생 산출물(reports·figures·notebooks)이 **없을 때만** 만든다.

    감사(2026-09-27) 반영 — 이전엔 이 함수가 매 테스트 전에 run_all·record_gates 를 **항상** 돌렸다.
    그러면 커밋된 산출물이 잘못돼도 테스트가 스스로 다시 만들어 통과해버린다(자가치유) → '커밋된 산출물의
    정합성' 이라는 검증 대상이 사라진다. 그래서 기본 동작을 이렇게 바꾼다:
    - reports/gates.json 이 있으면 **재생성하지 않는다**(커밋된 산출물을 그대로 검증한다).
    - 없으면(클론 직후·삭제됨) 만든다 — 이전 교훈(산출물 gitignore + 존재검사만 하면 클론에서 tests 가 붉어진다,
      bootstrap 을 test_self_check 안에만 두면 gate 테스트가 모른다)은 그대로 지킨다.
    - `MISSION_BOOTSTRAP=1 pytest` 로 주면 항상 재생성한다(코드 수정 후 의도적으로 쓸 것).
    """
    have_data_gen = (ROOT / "src" / "data_gen.py").exists()
    have_run_all = (ROOT / "scripts" / "run_all.py").exists()
    if not (have_data_gen or have_run_all):
        return                                   # 미구현 레포: skip 판정은 needs() 가 한다
    figs_ok = (ROOT / "figures").exists() and any((ROOT / "figures").glob("*.png"))
    metrics_ok = (ROOT / "reports" / "metrics.json").exists()
    gates_ok = (ROOT / "reports" / "gates.json").exists()
    if gates_ok and not BOOTSTRAP:
        return                                   # 커밋된 산출물을 검증 대상으로 삼는다(재생성하지 않는다)
    if have_run_all and (BOOTSTRAP or not (figs_ok and metrics_ok)):
        subprocess.run([sys.executable, "scripts/run_all.py"], cwd=ROOT, check=True,
                       capture_output=True, text=True)
    if have_data_gen:
        if not (ROOT / "data" / "products.csv").exists():
            subprocess.run([sys.executable, "src/data_gen.py", "--seed", "42"], cwd=ROOT, check=True,
                           capture_output=True, text=True)
        if (ROOT / "scripts" / "run_analysis.py").exists() and not (metrics_ok and figs_ok):
            subprocess.run([sys.executable, "scripts/run_analysis.py"], cwd=ROOT, check=True,
                           capture_output=True, text=True)
        if (ROOT / "scripts" / "spec_curve.py").exists() and not (ROOT / "reports" / "spec_curve.csv").exists():
            subprocess.run([sys.executable, "scripts/spec_curve.py"], cwd=ROOT, check=True,
                           capture_output=True, text=True)
    nb_ok = (ROOT / "notebooks").exists() and any((ROOT / "notebooks").glob("*.ipynb"))
    if not nb_ok and (ROOT / "scripts" / "build_notebook.py").exists():
        subprocess.run([sys.executable, "scripts/build_notebook.py"], cwd=ROOT, check=True,
                       capture_output=True, text=True)
    if (ROOT / "scripts" / "record_gates.py").exists() and (BOOTSTRAP or not gates_ok):
        subprocess.run([sys.executable, "scripts/record_gates.py"], cwd=ROOT, check=True,
                       capture_output=True, text=True)


@pytest.fixture(autouse=True)
def _bootstrap():
    ensure_inputs()


def gate_record(gid: str):
    """게이트 판정은 '이 레포가 **재서**한 값'으로만 내린다.

    reports/gates.json 형식:  {"G1": {"value": "IoU 52.4%", "pass": true, "how": "evaluate.py"}}
    이 파일이 없거나 항목이 없으면 실패다. '검사했는데 아무것도 말하지 않는' 상태를 막는 계약이다.
    """
    import json
    f = ROOT / "reports" / "gates.json"
    if not f.exists():
        raise AssertionError(f"{gid}: reports/gates.json 이 없다 → 게이트별 실측값을 먼저 기록하라")
    d = json.loads(f.read_text(encoding="utf-8"))
    if gid not in d:
        raise AssertionError(f"{gid}: gates.json 에 항목이 없다(키 목록: {sorted(d)})")
    rec = d[gid]
    assert set(("value", "pass")) <= set(rec), f"{gid}: {{'value':…,'pass':…}} 형식이어야 한다"
    assert rec["pass"] is True, f"{gid}: 통과 실패 — 측정값 {rec['value']}"
    return rec


def needs(*rel: str):
    """게이트의 증거 파일 기준 마크.

    - 미구현 + 기본        → skip  (초록이지만 README가 skip 수를 말한다)
    - 미구현 + STRICT      → 실행 → 본문 assert 가 **실패**한다(미구현을 남기지 않음)
    - 구현됨 + 증거 없음   → skip / STRICT이면 실패
    - 구현됨 + 증거 있음   → 실행
    """
    missing = [r for r in rel if not (ROOT / r).exists()]
    if not implemented():
        return pytest.mark.skipif(not STRICT, reason="미션 미구현 — 구현 후 실행")
    if missing:
        return pytest.mark.skipif(not STRICT, reason="산출물 없음: " + ", ".join(missing))
    return pytest.mark.skipif(False, reason="")
