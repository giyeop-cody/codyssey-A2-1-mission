"""수치 검사의 핵심 불변량. **실패하도록 설계**돼 있다(값을 바꾸거나 표를 지우면 붉어진다)."""
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def rep(name):
    f = ROOT / "reports" / name
    if not f.exists():
        pytest.skip(f"{name} 없음 — python scripts/run_all.py 를 먼저 실행")
    return json.loads(f.read_text(encoding="utf-8"))


def test_det_area_ratio_실측과_부합():
    la = rep("linear_algebra.json")
    worst = max(float(r["절대오차"]) for r in la["표"])
    assert worst <= 1e-9, f"면적비 vs |det| 오차 {worst:.2e} > 1e-9"
    assert any(r["검사가무의미"] for r in la["표"]), "det=1 행렬의 항등식 여부를 표가 표시해야 한다"


def test_power_iteration_단조():
    pi = rep("power_iteration.json")
    its = [r["반복수"] for r in pi["표"][:3]]
    assert its == sorted(its, reverse=True), f"반복수가 비(λ2/λ1)를 따라 줄지 않는다: {its}"
    assert all(float(r["상대오차%"]) <= 5 for r in pi["표"])


def test_svd_저장비율_역효과노출():
    sv = rep("svd.json")
    assert any(r["저장비율_%"] > 100 for r in sv["표"]), "k 큰 구간의 역효과를 감추면 안 된다"


def test_central_diff_검사에게_구분력이_있다():
    ca = rep("calculus.json")
    by = {r["함수"]: r for r in ca["표"]}
    assert by["x^2"]["1e-4_전구간통과"] is True, "x^2 은 통과해야 한다(그래야 '요구 1e-4'가 무의미함이 드러난다)"
    assert by["x^3"]["1e-4_전구간통과"] is False, "x^3 에서도 통과한다면 검사 자체가 무너진 것이다"


def test_gradcheck_상대오차_정의():
    """분모가 0인 항은 상대오차로 판정할 수 없다 → 정의를 명시한 값만 본다(A6-1 의 함정과 같다)."""
    bp = rep("backprop.json")
    for y in ("1.0", "0.0"):
        v = bp["y_true_별"][y]
        assert float(v["최대상대오차_0제외"]) <= 1e-7, f"y={y} 상대오차 {v['최대상대오차_0제외']}"
        assert float(v["최대절대오차"]) <= 1e-8, f"y={y} 0인 항의 절대오차 {v['최대절대오차']}"


def test_mle_동치는_격자해상도_안에서():
    pr = rep("probability.json")
    assert float(pr["MSE_MLE_동치"]["차이"]) == 0.0
    assert float(pr["CE_MLE_동치"]["차이"]) == 0.0
    assert pr["CE_MLE_동치"]["깁스_성립"] is True
    assert pr["softmax"]["차감버전_유한"] and not pr["softmax"]["naive_유한"]


def test_모멘텀_생존():
    op = rep("optimizer.json")
    tae = [r for r in op["표"] if r["함수"].startswith("타원") and r["lr"] == 0.15]
    van = next(r for r in tae if r["옵티"] == "vanilla")
    mom = next(r for r in tae if r["옵티"] == "momentum")
    assert van["판정"] == "발산" and float(mom["f_최종"]) < 1.0
