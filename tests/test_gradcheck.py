"""A2-1-T10: 손으로 쓴 역전파를 격자 독립적인 수치미분으로 검증 (허용 ≤1e-7)."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.mark.parametrize("y", [1.0, 0.0])
def test_backward_matches_numeric(y):
    import backprop as B
    h, n = B.backward(y), B.numeric_grad(y)
    for key in ("dW1", "dW2", "db1", "db2"):
        e = B.grad_errs(h[key], n[key])
        assert e["상대오차(0이_아닌_항)"] <= 1e-7, f"{key} 상대오차 {e['상대오차(0이_아닌_항)']:.2e} (y={y})"
        assert e["절대오차(전체)"] <= 1e-8, f"{key} 절대오차 {e['절대오차(전체)']:.2e} (y={y})"


def test_기울기에_0이_아닌_항이_존재한다():
    """전부 0이어도 '일치'하는 검사(항등식)를 막는다. x=[1,0] 때문에 W1 열 하나는 0이어 **정상**."""
    import backprop as B
    d = B.backward(1.0)["dW1"]
    assert d.shape == (2, 2)
    assert d[:, 0].any(), "첫 번째 열은 0이 아니어야 한다(x의 첫 성분이 1 이므로)"
    assert not d[:, 1].any(), "두 번째 열은 0이어야 한다(x의 둘째 성분이 0 이므로) — 다르면 forward 가 꼬인 것"
