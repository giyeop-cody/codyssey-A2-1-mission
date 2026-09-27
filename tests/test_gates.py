"""A2-1 통과 게이트 테스트.

감사(2026-09-27) 반영: 이전 4개 테스트는 **assert 가 하나도 없었다**(docstring + gate_record() 호출뿐) —
판정은 gates.json 의 자가 선언 필드에서 왔다. 이제 각 테스트가
(1) `gate_record` 로 선언 통과를 확인하고, (2) **reports/*.json 의 값을 직접 다시 판정**한다.
둘이 어긋나면(선언은 pass 인데 값이 조건을 못 채우면) 붉어진다.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import needs  # noqa: E402


def metric(name):
    f = ROOT / "reports" / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


@needs()
def test_01_G1_손계산_vs_NumPy_4자리_그리고_원문35_필수6항():
    """[G1] 원문 3-4/3-5/3-8: 손계산 ↔ NumPy ↔ 수치미분 4자리 일치 + 필수 유도 6항 기록."""
    from conftest import gate_record
    gate_record("G1")
    bp = metric("backprop.json")
    assert bp is not None, "reports/backprop.json 없음"
    keys = [r["키"] for r in bp["원문3-5_필수유도"]["표"]]
    assert keys == ["dy_pred", "dz2", "dW2", "da1", "dz1", "dW1"], f"원문 3-5 필수 6항 순서·구성: {keys}"
    for y in ("1.0", "0.0"):
        v = bp["y_true_별"][y]
        assert v["gate_4자리일치"] and v["gate_1e-7"], f"y={y} 게이트 실패: {v['최대상대오차_0제외']}"
        assert abs(v["yhat"] - 0.647292) < 1e-6, f"y={y} ŷ 가 고정예제 값과 다르다: {v['yhat']}"
        vals = bp["원문3-5_필수유도"]["y_true_별_값"][y]
        assert all(k in vals for k in keys), f"y={y} 6항 값 누락: {sorted(vals)}"
        # ∂L/∂z₂ = ŷ − y 와 ∂L/∂y_pred·ŷ(1−ŷ) 가 같은 값이어야 한다(연쇄법칙 일관성)
        assert abs(float(vals["dz2"]) - (v["yhat"] - float(y))) < 1e-5
        assert abs(float(vals["dz2"]) - float(vals["dy_pred"]) * v["yhat"] * (1 - v["yhat"])) < 1e-5


@needs()
def test_02_G2_원문21_x3_오차_그리고_등고선_직교():
    """[G2] 원문 2-1(x=3, 해석값 6, ≤1e-4) · 2-2(등고선 x²+y²) · 2-4(직교) + 항등식 표기."""
    from conftest import gate_record
    gate_record("G2")
    ca = metric("calculus.json")
    assert ca is not None, "reports/calculus.json 없음"
    spec = ca["원문2-1_검증"]
    assert "x=3" in spec["지점"], f"원문은 x=3 을 지정한다: {spec['지점']}"
    assert spec["해석값"] == 6.0, f"원문 해석값 6: {spec['해석값']}"
    assert float(spec["오차_최대(전h)"]) <= 1e-4, f"원문 2-1 오차 조건: {spec['오차_최대(전h)']}"
    perp = ca["직교성"]["원문: f = x^2 + y^2"]
    assert perp["표본수"] > 100 and float(perp["실측_최대편차_deg"]) <= 1.0, f"등고선 직교 편차: {perp}"
    # '검사가 항등식이다' 라는 판독도 사실이어야 한다(둘 다 참일 때만 G2 가 성립)
    by = {r["함수"]: r for r in ca["표"]}
    assert by["x^2"]["1e-4_전구간통과"] is True and by["x^3"]["1e-4_전구간통과"] is False
    assert float(perp["유도_최대편차_deg"]) == 0.0 and float(perp["실측_최대편차_deg"]) > 0.0


@needs()
def test_03_G3_원문42_반경_그리고_모멘텀_이점():
    """[G3] 원문 4-2(lr=0.1·100회·반경 0.1) · 4-6/4-7(모멘텀 이점) · 측정 상한 ≤ 이론 상한."""
    from conftest import gate_record
    gate_record("G3")
    op = metric("optimizer.json")
    assert op is not None, "reports/optimizer.json 없음"
    g42 = op["gate_원문4-2"]
    assert g42["gate_원문4-2"] and float(g42["반경_최종"]) <= 0.1, f"원문 4-2: {g42}"
    tae = [r for r in op["표"] if r["함수"].startswith("타원") and r["lr"] == 0.15]
    van = next(r for r in tae if r["옵티"] == "vanilla")
    mom = next(r for r in tae if r["옵티"] == "momentum")
    assert van["판정"] == "발산", f"타원 lr=0.15 vanilla 가 발산해야 모멘텀과 대비된다: {van}"
    assert mom["원문4-2_기준(반경≤0.1)"], f"타원 lr=0.15 momentum 반경: {mom['반경_최종']}"
    assert mom["판정"] != "발산"
    # 이론 상한: vanilla 는 2/λmax, heavy-ball 모멘텀은 2(1−β)/λmax 여야 한다(이전엔 둘 다 2/λmax 였다)
    ell = {k: v for k, v in op["수렴_상한"].items() if k.startswith("타원")}
    assert abs(ell["타원: x^2 + 10y^2 / vanilla"]["이론_상한"] - 0.1) < 1e-9
    assert abs(ell["타원: x^2 + 10y^2 / momentum"]["이론_상한"] - 0.01) < 1e-9, ell["타원: x^2 + 10y^2 / momentum"]
    for k, v in op["수렴_상한"].items():
        if v["측정_상한(자체기준)"] is not None and v["이론_상한"] is not None:
            assert v["측정_상한(자체기준)"] <= v["이론_상한"] + 1e-12, f"{k}: {v}"
    # 원문 4-4 는 원(x²+y²)에서 성립하지 않는다 — 그 대조 표가 있어야 한다(숨기지 않기)
    assert "원문4-4_대조(lr≥0.5 발산)" in op
    row = next(r for r in op["원문4-4_대조(lr≥0.5 발산)"]["표"] if r["lr"] == 0.5)
    assert "발산" not in row["원_vanilla"], f"원 lr=0.5 는 발산이 아니다(원문과 실측의 충돌): {row}"
    # 판정 라벨은 관측과 일치해야 한다: '진동'은 부호반전이 있을 때만
    for r in op["표"]:
        if r["판정"] == "정체(천천히 감소)":
            assert float(r["f_최종"]) >= 1e-8, r


@needs()
def test_04_G4_원문조항_게이트가_실제로_있다():
    """[G4] 원문 조항 게이트(1-4·1-6·1-7·3-5·5-2·5-3)가 리포트에 존재·통과 + 기대-실측 병기."""
    from conftest import gate_record
    gate_record("G4")
    la, pi, sv, pr = metric("linear_algebra.json"), metric("power_iteration.json"), metric("svd.json"), metric("probability.json")
    bp, op, ca = metric("backprop.json"), metric("optimizer.json"), metric("calculus.json")
    assert la and la["gate_원문1-4"]["통과"], f"원문 1-4: {la and la['gate_원문1-4']}"
    assert pi and pi["gate_원문1-6"]["통과"], f"원문 1-6: {pi and pi['gate_원문1-6']}"
    assert sv and set(sv["gate_원문1-7"]["요구_k"]) <= set(sv["gate_원문1-7"]["그림_k"]), sv and sv["gate_원문1-7"]
    assert bp and len(bp["원문3-5_필수유도"]["표"]) == 6
    assert pr and all(abs(float(v["합"]) - 1.0) <= 1e-9 for v in pr["베르누이PMF(원문5-2)"].values()), \
        f"원문 5-2: 베르누이 PMF 합이 1 이어야 한다: {pr and pr['베르누이PMF(원문5-2)']}"
    assert pr and pr["softmax"]["gate_합=1_±1e-6"], f"원문 5-3: {pr and pr['softmax']}"
    assert op and op["gate_원문4-2"]["gate_원문4-2"]
    assert ca and ca["원문2-1_검증"]["gate_≤1e-4"]
    analysis = (ROOT / "reports" / "analysis.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "격자" in analysis and "반복" in analysis
    assert "기대" in readme and "실측" in readme
    assert ("스펙 이탈" in readme or "정정" in readme), "README 에 스펙 이탈·정정 기록이 있어야 한다"


@needs()
def test_05_원문_지정_분포와_함수가_실제로_쓰였다():
    """[추가] 산출물이 원문 지정 파라미터(N(2,0.5)·B(0.3)/B(0.7)·x=3·x²+y²)로 만들어졌는지 값으로 확인한다."""
    pr, ca = metric("probability.json"), metric("calculus.json")
    assert pr is not None and ca is not None
    normals = pr["정규PDF(원문5-1)"]
    assert set(normals) == {"N(0,1)", "N(2,0.5)"}, f"원문 5-1 분포: {sorted(normals)}"
    assert normals["N(2,0.5)"]["mu"] == 2.0 and normals["N(2,0.5)"]["sigma"] == 0.5
    assert abs(normals["N(2,0.5)"]["peak"] - 0.797885) < 1e-5, normals["N(2,0.5)"]["peak"]
    bern = pr["베르누이PMF(원문5-2)"]
    assert set(bern) == {"B(0.3)", "B(0.7)"} and bern["B(0.3)"]["PMF_k0_k1"] == [0.7, 0.3]
    assert "x=3" in ca["원문2-1_검증"]["지점"]
    assert "원문: f = x^2 + y^2" in ca["직교성"]
    assert "원문 함수 x²+y²" in ca["조건"] or "x²+y²" in ca["조건"], ca["조건"]
    assert "추가(타원)" in str(ca["직교성"]), "타원 x²+4y² 는 '추가 대조'로만 남아야 한다"
    assert (ROOT / "figures" / "08_power_iteration.png").exists(), "원문 1-5/1-6 시각화 그림이 없다"
