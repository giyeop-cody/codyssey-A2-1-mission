"""A2-1 요구 조항 자가채점 — 통과 못 하면 exit 1 (푸시·제출 전 게이트)."""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REP, FIG, NB = ROOT / "reports", ROOT / "figures", ROOT / "notebooks"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def jload(name: str):
    p = REP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    la, pi, svd = jload("linear_algebra.json"), jload("power_iteration.json"), jload("svd.json")
    ca, op, bp, pr = jload("calculus.json"), jload("optimizer.json"), jload("backprop.json"), jload("probability.json")

    # 1. 선형대수
    check("src/linear_algebra.py 존재", (ROOT / "src/linear_algebra.py").exists())
    check("그림 01 변환 전후 겹치기", (FIG / "01_linalg_transforms.png").exists())
    if la:
        rows = la["표"]
        check("det·면적비 표 4행↑", len(rows) >= 4, f"{len(rows)}행")
        check("면적비 오차 ≤1e-9", max(float(r["절대오차"]) for r in rows) <= 1e-9,
              f"최대 {max(float(r['절대오차']) for r in rows):.2e}")
        check("'무의미' 검사 여부를 표가 스스로 표시", any(r["검사가무의미"] for r in rows),
              f"항등식 행 {sum(1 for r in rows if r['검사가무의미'])}개")
        check("det=1 행에 대안 검사(축비)가 있다", any("축비" in r["행렬"] for r in rows))
    # 2. power iteration
    if pi:
        its = [r["반복수"] for r in pi["표"][:3]]
        check("power iteration 3케이스", len(its) == 3, str(its))
        check("비 ↓ → 반복수 ↓ (단조)", its[0] > its[1] > its[2], " → ".join(map(str, its)))
        check("λ 상대오차 ≤5%", all(float(r["상대오차%"]) <= 5.0 for r in pi["표"]),
              f"최대 {max(float(r['상대오차%']) for r in pi['표']):.2e}%")
        check("v0 직교 케이스가 표에 있다", "직교" in json.dumps(pi, ensure_ascii=False))
    # 3. SVD
    if svd:
        ks = [r["k"] for r in svd["표"]]
        check("SVD k=1·10·50 포함", {1, 10, 50} <= set(ks), str(ks))
        check("저장비율 >100% 구간 노출(압축 역효과)", any(r["저장비율_%"] > 100 for r in svd["표"]))
        check("k≥랭크에서 개선 0(중복 노출)", any(r["k"] == 100 and r["실제 랭크"] == 64 for r in svd["표"]))
    # 4. 미적분
    if ca:
        by = {r["함수"]: r for r in ca["표"]}
        check("수치미분 표 3함수 × 8h", len(ca["표"]) >= 3, f"{len(ca['표'])}행")
        check("x^2 은 전 h 통과(=항등식임이 기록됨)", by["x^2"]["1e-4_전구간통과"] is True)
        check("x^3·sin 은 통과 실패(=검사에게 구분력이 있다)",
              by["x^3"]["1e-4_전구간통과"] is False or by["sin(x)"]["1e-4_전구간통과"] is False)
        check("최적 h 가 함수마다 다르다", len({by[k]["h_최적"] for k in by}) >= 2,
              str({k: by[k]["h_최적"] for k in by}))
        check("그림 03(등고선+그라디언트)", (FIG / "03_gradient_contours.png").exists())
        check("직교 검사: 유도 접선은 0° 편차(항등식)로 표시",
              float(ca["직교성"]["유도_최대편차_deg"]) == 0.0 and float(ca["직교성"]["실측_최대편차_deg"]) > 0,
              f"유도 0 / 실측 {ca['직교성']['실측_최대편차_deg']}°")
    # 5. 최적화
    if op:
        rows = op["표"]
        check("lr 스윕 2함수×3옵티×11lr 이상", len(rows) >= 66, f"{len(rows)}행")
        check("발산과 수렴이 모두 관측", any(r["판정"] == "발산" for r in rows) and any(r["판정"] == "수렴" for r in rows))
        tae = [r for r in rows if r["함수"].startswith("타원") and r["lr"] == 0.15]
        van = [r for r in tae if r["옵티"] == "vanilla"][0]
        mom = [r for r in tae if r["옵티"] == "momentum"][0]
        check("타원 lr=0.15: vanilla 발산 · momentum 생존",
              van["판정"] == "발산" and float(mom["f_최종"]) < 1.0, f"{van['f_최종']} / {mom['f_최종']}")
        check("측정 상한 ≤ 이론 상한", all(v["측정_상한"] in (None, "None") or v["측정_상한"] <= v["이론_상한"] + 1e-12
                                    for v in op["수렴_상한"].values()),
              str({k: (v["측정_상한"], v["이론_상한"]) for k, v in op["수렴_상한"].items() if v["측정_상한"]}))
        check("반복수 민감도 표(판정 기준이 비교 대상)", len(op.get("반복수_민감도", [])) >= 4)
        check("그림 04 경로 겹치기·05 lr 스윕", (FIG / "04_optimizer_paths.png").exists()
              and (FIG / "05_lr_sweep.png").exists())
        check("lr 단독 주장 금지 문구가 리포트에 있다", "함수·기준·반복수를 함께" in op["주의"])
    # 6. 역전파
    if bp:
        check("손계산 예제 값 확보(ŷ=0.647292)", abs(bp["고정예제"]["yhat"] - 0.647292) < 1e-6)
        for y in ("1.0", "0.0"):
            v = bp["y_true_별"][y]
            check(f"y_true={y}: 4자리 일치", v["gate_4자리일치"], f"dW1 {v['dW1']}")
            check(f"y_true={y}: gradcheck ≤1e-7(0항 제외)", v["gate_1e-7"], v["최대상대오차_0제외"])
        check("x=[1,0] 로 죽는 열을 '결핍'으로 설명", bp["기울기_0_열"]["개수"] == 1)
        check("y_true 미정의를 양쪽 계산으로 대응", "양쪽" in bp["y_true_미정의_대응"])
        check("관례(전치 함정) 명시", "outer" in bp["관례"])
    # 7. 확률
    if pr:
        check("MSE 최소 = 로그우도 최대", float(pr["MSE_MLE_동치"]["차이"]) == 0.0,
              pr["MSE_MLE_동치"]["차이"])
        check("CE 최소 = MLE", float(pr["CE_MLE_동치"]["차이"]) == 0.0, pr["CE_MLE_동치"]["차이"])
        check("깁스 부등식은 참값 기준으로 성립", pr["CE_MLE_동치"]["깁스_성립"] is True,
              f"CE {pr['CE_MLE_동치']['참값대_ fitted_교차엔트로피']} ≥ H {pr['CE_MLE_동치']['진_엔트로피']}")
        check("플러그인 편향을 숫자로 적었다", "플러그인_편향" in pr["CE_MLE_동치"],
              str(pr["CE_MLE_동치"].get("플러그인_편향")))
        check("softmax: 큰 스케일에서만 안정화가 검증된다",
              pr["softmax"]["차감버전_유한"] and not pr["softmax"]["naive_유한"])
        check("정규 PDF 넓이는 구간 의존임을 적었다", "적분 구간" in pr["정규PDF"]["넓이_주의"])
    # 8. 노트북·문서
    for nb_name, need in [("backprop_derivation.ipynb", "(2,2)"), ("probability_loss.ipynb", "MLE")]:
        f = NB / nb_name
        check(f"{nb_name} 존재·실행결과·'{need}'",
              f.exists() and '"outputs"' in f.read_text(encoding="utf-8") and need in f.read_text(encoding="utf-8"),
              "" if f.exists() else "파일 없음")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    check("README '요약' 섹션", "주요 결과 요약" in readme)
    check("README 에 5조항 검수표", readme.count("검증") >= 3 and "요구" in readme)
    check("README 에 갭 표", "갭" in readme)
    analysis = (REP / "analysis.md").read_text(encoding="utf-8") if (REP / "analysis.md").exists() else ""
    check("analysis.md 가 격자 해상도 의존을 명시", "격자" in analysis)
    check("analysis.md 에 그림 참조", analysis.count("figures/") >= 3 or "그림" in analysis)

    w = max(len(n) for n, _, _ in results) + 1
    npass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{w}} {detail}")
    print(f"---- {npass}/{len(results)} PASS")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
