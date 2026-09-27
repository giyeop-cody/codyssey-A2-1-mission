"""수치 검사의 핵심 불변량. **실패하도록 설계**돼 있다(값을 바꾸거나 표를 지우면 붉어진다).

감사(2026-09-27) 반영: 원문 조항 값(분포 파라미터·PMF 합·x=3·등고선 함수·softmax 합)에 대한 불변량을 추가하고,
키 이름 변경(λ_참값 → λ_참값_eig 등)을 반영했다.
"""
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
    area_rows = [r for r in la["표"] if "면적비(실측)" in r]
    worst = max(float(r["절대오차"]) for r in area_rows)
    assert worst <= 1e-9, f"면적비 vs |det| 오차 {worst:.2e} > 1e-9"
    assert any(r["검사가무의미"] for r in la["표"]), "det=1 행렬의 항등식 여부를 표가 표시해야 한다"
    # 축비 값은 '면적비' 열에 섞이면 안 된다(열 이름 = 측정 정의)
    assert all(abs(float(r["면적비(실측)"]) - abs(float(r["det"]))) <= 1e-9 for r in area_rows), \
        "면적비 열에 면적비가 아닌 값(축비 등)이 들어 있다"
    axis_row = [r for r in la["표"] if "축비(실측)" in r]
    assert axis_row and abs(float(axis_row[0]["축비(실측)"]) - 4.0) <= 1e-6, "S(2,0.5) 축비 4.0 은 별도 열에 있어야 한다"
    assert la["gate_원문1-4"]["통과"] is True


def test_power_iteration_단조():
    pi = rep("power_iteration.json")
    its = [r["반복수"] for r in pi["표"][:3]]
    assert its == sorted(its, reverse=True), f"반복수가 비(λ2/λ1)를 따라 줄지 않는다: {its}"
    # 원문 1-6: 비교 대상은 np.linalg.eig, 오차 ≤5%
    assert all(float(r["상대오차%(vs eig)"]) <= 5 for r in pi["표"][:3])
    assert all(r["gate_원문1-6(≤5%)"] for r in pi["표"][:3])
    assert all(abs(r["λ_참값_eig"] - r["λ_참값_eigvalsh"]) <= 1e-6 for r in pi["표"]), \
        "대칭행렬에서 eig 와 eigvalsh 가 다르면 어딘가 틀린 것이다"
    # 이론 예측은 실제 수렴 기준(Rayleigh 변화)과 같은 양이어야 한다 → 실측과 자릿수가 맞아야 한다
    for r in pi["표"][:3]:
        assert 0.5 <= r["반복수"] / max(r["이론_예측반복_Rayleigh"], 1) <= 2.0, \
            f"예측 {r['이론_예측반복_Rayleigh']} vs 실측 {r['반복수']} — 기준이 다른 공식이다"
    # 마지막 행(직교 v0)은 λ_max 를 못 찾은 **실패**여야 한다
    last = pi["표"][-1]
    assert float(last["상대오차%(vs eig)"]) > 5.0 and "실패" in last["판정"], \
        f"직교 v0 케이스는 실패 사례로 기록돼야 한다: {last}"


def test_svd_저장비율_역효과노출():
    sv = rep("svd.json")
    assert any(r["저장비율_%"] > 100 for r in sv["표"]), "k 큰 구간의 역효과를 감추면 안 된다"
    # 원문 1-7: k=10·50·100 이 그림에 있어야 한다
    assert set(sv["gate_원문1-7"]["요구_k"]) <= set(sv["gate_원문1-7"]["그림_k"])
    k100 = next(r for r in sv["표"] if r["k"] == 100)
    k64 = next(r for r in sv["표"] if r["k"] == 64)
    assert k100["mse"] == k64["mse"] and k100["실제 랭크"] == 64, "k=100 은 랭크 상한이라 k=64 와 같아야 한다"


def test_central_diff_검사에게_구분력이_있다():
    ca = rep("calculus.json")
    by = {r["함수"]: r for r in ca["표"]}
    # 원문 2-1: x=3 에서 해석값 6 과 비교, 오차 ≤1e-4
    spec = ca["원문2-1_검증"]
    assert "x=3" in spec["지점"] and spec["해석값"] == 6.0
    assert float(spec["오차_최대(전h)"]) <= 1e-4 and spec["gate_≤1e-4"] is True
    assert all(r["x"] == 3.0 for r in ca["표"]), "표의 기본 지점은 원문대로 x=3 이어야 한다"
    assert by["x^2"]["1e-4_전구간통과"] is True, "x^2 은 통과해야 한다(그래야 '요구 1e-4'가 무의미함이 드러난다)"
    assert by["x^3"]["1e-4_전구간통과"] is False, "x^3 에서도 통과한다면 검사 자체가 무너진 것이다"
    # 원문 2-2: 등고선은 x²+y²
    perp = ca["직교성"]["원문: f = x^2 + y^2"]
    assert float(perp["실측_최대편차_deg"]) <= 1.0 and perp["표본수"] > 100
    assert float(perp["유도_최대편차_deg"]) == 0.0, "유도 접선이 0° 가 아니면 구현이 틀린 것이다(항등식)"


def test_gradcheck_상대오차_정의():
    """분모가 0인 항은 상대오차로 판정할 수 없다 → 정의를 명시한 값만 본다(A6-1 의 함정과 같다)."""
    bp = rep("backprop.json")
    for y in ("1.0", "0.0"):
        v = bp["y_true_별"][y]
        assert float(v["최대상대오차_0제외"]) <= 1e-7, f"y={y} 상대오차 {v['최대상대오차_0제외']}"
        assert float(v["최대절대오차"]) <= 1e-8, f"y={y} 0인 항의 절대오차 {v['최대절대오차']}"


def test_원문35_필수유도_6항이_다_있다():
    bp = rep("backprop.json")
    keys = [r["키"] for r in bp["원문3-5_필수유도"]["표"]]
    assert keys == ["dy_pred", "dz2", "dW2", "da1", "dz1", "dW1"], keys
    for y in ("1.0", "0.0"):
        vals = bp["원문3-5_필수유도"]["y_true_별_값"][y]
        assert set(keys) <= set(vals)
        # 연쇄법칙 일관성: ∂L/∂z₂ = ∂L/∂y_pred·ŷ(1−ŷ) = ŷ − y
        yhat = bp["y_true_별"][y]["yhat"]
        assert abs(float(vals["dz2"]) - float(vals["dy_pred"]) * yhat * (1 - yhat)) < 1e-5
        assert abs(float(vals["dz2"]) - (yhat - float(y))) < 1e-5
        # ∂L/∂a₁ = W₂·∂L/∂z₂ (W₂=[0.5,0.6])
        assert np.allclose(vals["da1"], np.array([0.5, 0.6]) * float(vals["dz2"]), atol=1e-5)
        # ∂L/∂W₂ = a₁·∂L/∂z₂ (a₁ = [0.524979, 0.574443])
        assert np.allclose(vals["dW2"], np.array([0.524979, 0.574443]) * float(vals["dz2"]), atol=1e-5)


def test_mle_동치는_격자해상도_안에서():
    pr = rep("probability.json")
    assert float(pr["MSE_MLE_동치"]["차이"]) == 0.0
    assert float(pr["CE_MLE_동치"]["차이"]) == 0.0
    assert pr["CE_MLE_동치"]["깁스_성립"] is True
    assert pr["softmax"]["차감버전_유한"] and not pr["softmax"]["naive_유한"]


def test_원문5_분포와_softmax합():
    pr = rep("probability.json")
    # 원문 5-1: N(0,1) 과 N(2,0.5)
    normals = pr["정규PDF(원문5-1)"]
    assert set(normals) == {"N(0,1)", "N(2,0.5)"}
    assert normals["N(2,0.5)"]["mu"] == 2.0 and normals["N(2,0.5)"]["sigma"] == 0.5
    assert abs(normals["N(2,0.5)"]["peak"] - 0.797885) < 1e-5
    assert abs(normals["N(0,1)"]["peak"] - 0.398942) < 1e-5
    assert normals["N(2,0.5)"]["적분[-40,40]"] > 0.999999
    # 원문 5-2: 베르누이 PMF 는 k∈{0,1} 두 값, 합 1
    bern = pr["베르누이PMF(원문5-2)"]
    assert bern["B(0.3)"]["PMF_k0_k1"] == [0.7, 0.3] and bern["B(0.7)"]["PMF_k0_k1"] == [0.3, 0.7]
    for label, v in bern.items():
        assert abs(float(v["합"]) - 1.0) <= 1e-12, f"{label} PMF 합 {v['합']} ≠ 1"
    # 원문 5-3: softmax 합 = 1.0 ± 1e-6
    assert pr["softmax"]["gate_합=1_±1e-6"] is True
    assert float(pr["softmax"]["최대_합-1_절대오차"]) <= 1e-6


def test_모멘텀_생존():
    op = rep("optimizer.json")
    tae = [r for r in op["표"] if r["함수"].startswith("타원") and r["lr"] == 0.15]
    van = next(r for r in tae if r["옵티"] == "vanilla")
    mom = next(r for r in tae if r["옵티"] == "momentum")
    assert van["판정"] == "발산", van
    assert mom["판정"] != "발산" and mom["원문4-2_기준(반경≤0.1)"], mom
    assert float(mom["f_최종"]) < 1.0


def test_원문42_반경과_판정라벨_일관성():
    op = rep("optimizer.json")
    assert op["gate_원문4-2"]["gate_원문4-2"] is True
    assert float(op["gate_원문4-2"]["반경_최종"]) <= 0.1
    for r in op["표"]:
        f1 = float(r["f_최종"]) if r["f_최종"] != "inf" else float("inf")
        if r["판정"] == "수렴":
            assert f1 < 1e-8, r
        elif r["판정"] == "정체(천천히 감소)":
            assert f1 >= 1e-8 and f1 <= r["f_시작"], f"'정체' 라벨인데 감소하지 않았다: {r}"
        elif r["판정"] == "발산":
            assert (f1 > r["f_시작"]) or not np.isfinite(f1), r
    # 원문 4-4 대조: 원(x²+y²) 에서는 lr=0.5 가 발산이 아니다(원문과 실측의 충돌을 숨기지 않는다)
    row = next(r for r in op["원문4-4_대조(lr≥0.5 발산)"]["표"] if r["lr"] == 0.5)
    assert "발산" not in row["원_vanilla"] and "발산" in row["타원_vanilla"], row
