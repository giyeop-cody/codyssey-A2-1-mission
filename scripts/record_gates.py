"""reports/*.json → reports/gates.json. 게이트 '통과'는 사람이 선언하지 않고 여기서 계산된다."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REP = ROOT / "reports"


def j(name):
    p = REP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


bp, ca, op = j("backprop.json"), j("calculus.json"), j("optimizer.json")
gates = {}

if bp is None:
    gates["G1"] = {"value": "reports/backprop.json 없음 → 미실행", "pass": False, "how": "python scripts/run_all.py"}
else:
    a, b = bp["y_true_별"]["1.0"], bp["y_true_별"]["0.0"]
    gates["G1"] = {
        "value": (f"ŷ={bp['고정예제']['yhat']} · y=1: L={a['L']}, ∂L/∂z₂={a['dz2']}, ∂L/∂W₁={a['dW1']} "
                  f"(수치미분 최대상대오차 {a['최대상대오차_0제외']}) / y=0: L={b['L']}, ∂L/∂z₂={b['dz2']}"),
        "pass": bool(a["gate_4자리일치"] and b["gate_4자리일치"] and a["gate_1e-7"] and b["gate_1e-7"]),
        "how": "src/backprop.py:main() — 손계산 vs 독립 수치미분(격자 h=1e-5), 0인 항은 절대오차로 판정"}

if ca is None:
    gates["G2"] = {"value": "reports/calculus.json 없음 → 미실행", "pass": False, "how": "python scripts/run_all.py"}
else:
    by = {r["함수"]: r for r in ca["표"]}
    vac = by["x^2"]["1e-4_전구간통과"] is True
    discriminating = any(by[k]["1e-4_전구간통과"] is False for k in ("x^3", "sin(x)"))
    gates["G2"] = {
        "value": (f"항등식 표기: x^2 은 전 h ≤1e-4 통과({vac}) · 구분력: x^3/sin 실패({discriminating}) · "
                  f"직교성: 유도 접선 편차 {ca['직교성']['유도_최대편차_deg']}°(항등식) vs 그려진 등고선 "
                  f"{ca['직교성']['실측_최대편차_deg']}°(표본 {ca['직교성']['표본수']})"),
        "pass": bool(vac and discriminating and float(ca["직교성"]["유도_최대편차_deg"]) == 0.0
                     and float(ca["직교성"]["실측_최대편차_deg"]) > 0),
        "how": "src/calculus.py:main() — 요구 1e-4 조건이 어느 함수에서 죽는지 표로 남긴다"}

if op is None:
    gates["G3"] = {"value": "reports/optimizer.json 없음 → 미실행", "pass": False, "how": "python scripts/run_all.py"}
else:
    rows = op["표"]
    tae = [r for r in rows if r["함수"].startswith("타원") and r["lr"] == 0.15]
    van = next(r for r in tae if r["옵티"] == "vanilla")
    mom = next(r for r in tae if r["옵티"] == "momentum")
    up = {k: (v["측정_상한"], v["이론_상한"]) for k, v in op["수렴_상한"].items() if v["측정_상한"]}
    ok_bound = all(m_ <= t_ + 1e-12 for m_, t_ in up.values())
    gates["G3"] = {
        "value": (f"타원 lr=0.15: vanilla {van['f_최종']}({van['판정']}) / momentum {mom['f_최종']}({mom['판정']}) · "
                  f"수렴 상한 측정 vs 이론: " + ", ".join(f"{k.split(' / ')[0].split(':')[0]} {m}≤{t}" for k, (m, t) in up.items())),
        "pass": bool(van["판정"] == "발산" and float(mom["f_최종"]) < 1.0 and ok_bound),
        "how": "src/optimizer.py:main() — lr 13×함수 2×옵티 3 스윕, 반복수 100/1000/10000 민감도 포함"}

# G4: 계획(probe) 값과 구현 실측이 다른 항목을 표로 두었는가 — 문서 모양 게이트(강도 낮음, 갭 표에 한계를 적는다)
analysis = (REP / "analysis.md").read_text(encoding="utf-8") if (REP / "analysis.md").exists() else ""
readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
gates["G4"] = {
    "value": (f"리포트에 격자·반복수·tol 명시={'격자' in analysis and '반복' in analysis}, "
              f"README 에 기대-실측 병기={'기대' in readme or 'probe' in readme}"),
    "pass": ("격자" in analysis and ("기대" in readme or "probe" in readme)),
    "how": "scripts/run_all.py 가 쓰는 analysis.md·README 검증(문서 형태 검사 — 값 자체의 검증은 G1~G3 이 한다)",
}

out = REP / "gates.json"
out.write_text(json.dumps(gates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
for k, v in gates.items():
    print(f"[{'PASS' if v['pass'] else 'FAIL'}] {k} {v['value'][:104]}")
raise SystemExit(0 if all(v["pass"] for v in gates.values()) else 1)
