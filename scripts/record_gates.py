"""reports/*.json → reports/gates.json. 게이트 '통과'는 사람이 선언하지 않고 여기서 계산된다.

감사(2026-09-27) 반영: 게이트가 **미션 원문 조항**을 판정하도록 조건을 바꿨다.
이전 G2 는 "x²가 전 h 에서 통과한다(=요구가 무의미하다)"는 **이 레포의 서사**가 참이면 통과하는 구조였다
(결론이 통과조건인 순환). 이제 G2 는 원문 2-1(x=3, 해석값 6, ≤1e-4)·2-2(등고선 x²+y²)·2-4(직교)를 판정하고,
'무의미함'은 게이트가 아니라 리포트의 판독으로 남긴다. G3 에는 원문 4-2(반경 0.1)와 모멘텀 이론상한을 넣었다.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REP = ROOT / "reports"


def j(name):
    p = REP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


bp, ca, op = j("backprop.json"), j("calculus.json"), j("optimizer.json")
pi, sv, la, pr = j("power_iteration.json"), j("svd.json"), j("linear_algebra.json"), j("probability.json")
gates = {}

# ── G1: 원문 3-4/3-5/3-8 손계산 ↔ NumPy ↔ 수치미분 ─────────────────────────────
if bp is None:
    gates["G1"] = {"value": "reports/backprop.json 없음 → 미실행", "pass": False, "how": "python scripts/run_all.py"}
else:
    a, b = bp["y_true_별"]["1.0"], bp["y_true_별"]["0.0"]
    req_keys = [r["키"] for r in bp["원문3-5_필수유도"]["표"]]
    have = all(k in bp["원문3-5_필수유도"]["y_true_별_값"]["1.0"] for k in req_keys)
    gates["G1"] = {
        "value": (f"ŷ={bp['고정예제']['yhat']} · y=1: L={a['L']}, ∂L/∂y_pred={a['dy_pred(∂L/∂y_pred)']}, "
                  f"∂L/∂z₂={a['dz2']}, ∂L/∂a₁={a['da1(∂L/∂a₁)']}, ∂L/∂W₁={a['dW1']} "
                  f"(수치미분 최대상대오차 {a['최대상대오차_0제외']}) / y=0: L={b['L']}, ∂L/∂z₂={b['dz2']} · "
                  f"원문 3-5 필수 6항 기록={have}({req_keys})"),
        "pass": bool(a["gate_4자리일치"] and b["gate_4자리일치"] and a["gate_1e-7"] and b["gate_1e-7"] and have),
        "how": "src/backprop.py:main() — 손계산 vs 독립 수치미분(격자 h=1e-5), 0인 항은 절대오차로 판정"}

# ── G2: 원문 2-1(x=3, 해석값 6, ≤1e-4) · 2-2(등고선 x²+y²) · 2-4(직교) ──────────
if ca is None:
    gates["G2"] = {"value": "reports/calculus.json 없음 → 미실행", "pass": False, "how": "python scripts/run_all.py"}
else:
    spec = ca["원문2-1_검증"]
    perp = ca["직교성"]["원문: f = x^2 + y^2"]
    by = {r["함수"]: r for r in ca["표"]}
    x_ok = "x=3" in spec["지점"]
    err_ok = bool(spec["gate_≤1e-4"]) and float(spec["오차_최대(전h)"]) <= 1e-4
    contour_ok = perp["표본수"] > 100 and float(perp["실측_최대편차_deg"]) <= 1.0
    identity_flagged = by["x^2"]["1e-4_전구간통과"] is True and float(perp["유도_최대편차_deg"]) == 0.0
    discriminating = any(by[k]["1e-4_전구간통과"] is False for k in ("x^3", "sin(x)"))
    gates["G2"] = {
        "value": (f"원문 2-1: {spec['지점']} 해석값 {spec['해석값']} 오차 최대 {spec['오차_최대(전h)']} → ≤1e-4 {err_ok} · "
                  f"원문 2-2/2-4: 등고선 x²+y² 실측 직교 편차 최대 {perp['실측_최대편차_deg']}°"
                  f"(중위 {perp['실측_중위편차_deg']}°, {perp['표본수']}점) ≤1° {contour_ok} · "
                  f"항등식 표기(x² 전 h 통과 + 유도접선 0°) {identity_flagged} · 구분력(x³/sin 실패) {discriminating}"),
        "pass": bool(x_ok and err_ok and contour_ok and identity_flagged and discriminating),
        "how": "src/calculus.py:main() — 원문 조항을 직접 판정하고, '요구가 항등식이라 무의미하다'는 판독은 리포트에 남긴다"}

# ── G3: 원문 4-2(반경 0.1) · 4-4 대조 · 4-6/4-7(모멘텀) · 이론 상한 ──────────────
if op is None:
    gates["G3"] = {"value": "reports/optimizer.json 없음 → 미실행", "pass": False, "how": "python scripts/run_all.py"}
else:
    rows = op["표"]
    tae = [r for r in rows if r["함수"].startswith("타원") and r["lr"] == 0.15]
    van = next(r for r in tae if r["옵티"] == "vanilla")
    mom = next(r for r in tae if r["옵티"] == "momentum")
    up = {k: (v["측정_상한(자체기준)"], v["이론_상한"]) for k, v in op["수렴_상한"].items()
          if v["측정_상한(자체기준)"] is not None and v["이론_상한"] is not None}
    ok_bound = all(m_ <= t_ + 1e-12 for m_, t_ in up.values())
    long = [r for r in op["반복수_민감도"] if r["함수"].startswith("타원") and r["lr"] == 0.15][0]
    mom_long_ok = float(long["mom_1000"]) < 1e-8
    gates["G3"] = {
        "value": (f"원문 4-2: {op['gate_원문4-2']['원문']} → 반경 {op['gate_원문4-2']['반경_최종']} "
                  f"({op['gate_원문4-2']['gate_원문4-2']}) · 타원 lr=0.15: vanilla {van['f_최종']}({van['판정']}) / "
                  f"momentum {mom['f_최종']}({mom['판정']}, 원문기준 {mom['원문4-2_기준(반경≤0.1)']}) · 1000회 {long['mom_1000']} · "
                  "수렴 상한 측정 vs 이론: " + ", ".join(f"{k.split(' / ')[0].split(':')[0]} {m}≤{t}" for k, (m, t) in up.items())),
        "pass": bool(op["gate_원문4-2"]["gate_원문4-2"] and van["판정"] == "발산"
                     and mom["원문4-2_기준(반경≤0.1)"] and mom_long_ok and ok_bound),
        "how": "src/optimizer.py:main() — lr 13×함수 2×옵티 3 스윕 + 반복수 100/1000/10000 민감도 + 원문 4-4 대조 표"}

# ── G4: 원문 조항 게이트들이 리포트에 실제로 있는가 + 기대-실측 병기(문서 모양) ────────
analysis = (REP / "analysis.md").read_text(encoding="utf-8") if (REP / "analysis.md").exists() else ""
readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
spec_gates = {}
if la:
    spec_gates["원문1-4(면적비≤1%)"] = bool(la["gate_원문1-4"]["통과"])
if pi:
    spec_gates["원문1-6(eig비교≤5%)"] = bool(pi["gate_원문1-6"]["통과"])
if sv:
    spec_gates["원문1-7(그림 k=10·50·100)"] = set(sv["gate_원문1-7"]["요구_k"]) <= set(sv["gate_원문1-7"]["그림_k"])
if pr:
    spec_gates["원문5-2(PMF 합=1)"] = all(abs(float(v["합"]) - 1.0) <= 1e-9 for v in pr["베르누이PMF(원문5-2)"].values())
    spec_gates["원문5-3(softmax 합=1±1e-6)"] = bool(pr["softmax"]["gate_합=1_±1e-6"])
if bp is not None:
    spec_gates["원문3-5(필수유도 6항)"] = len(bp["원문3-5_필수유도"]["표"]) == 6
gates["G4"] = {
    "value": (f"원문 조항 게이트: {json.dumps(spec_gates, ensure_ascii=False)} · "
              f"리포트에 격자·반복수·tol 명시={'격자' in analysis and '반복' in analysis}, "
              f"README 에 기대-실측 병기={'기대' in readme and '실측' in readme}, "
              f"README 에 스펙 이탈·정정 기록={'스펙 이탈' in readme or '정정' in readme}"),
    "pass": bool(all(spec_gates.values()) and len(spec_gates) >= 6
                 and "격자" in analysis and "반복" in analysis
                 and "기대" in readme and "실측" in readme
                 and ("스펙 이탈" in readme or "정정" in readme)),
    "how": "scripts/run_all.py 가 쓰는 analysis.md·README 와 원문 조항 게이트들의 존재·통과 여부 "
           "(문서 형태 검사가 섞여 있다 — 강도 한계를 README 갭 표에 적는다)"}

(REP / "gates.json").write_text(json.dumps(gates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
for k, v in gates.items():
    print(f"{k}: {'PASS' if v['pass'] else 'FAIL'} — {v['value'][:180]}")
