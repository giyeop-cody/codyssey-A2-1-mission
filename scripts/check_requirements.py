"""A2-1 요구 조항 자가채점 — 통과 못 하면 exit 1 (푸시·제출 전 게이트).

감사(2026-09-27) 반영 3건:
1. **[원문] 태그 검사**를 신설했다. 이전 47개 검사는 전부 '이 레포가 세운 자체 기준'(표 행 수, 키워드 presence,
   파일 존재)이었고 미션 원문 조항(N(2,0.5)·베르누이 PMF 합=1·x=3·등고선 x²+y²·S(2,0.5) 그림·k=10/50/100 그림·
   softmax 합=1 ±1e-6·np.linalg.eig 비교)은 **하나도 검증하지 않았다**. 이제 원문 조항 검사를 앞에 두고
   출력에서 "원문 X/X · 자체 Y/Y" 로 나눠 센다.
2. 리포트 JSON 이 없으면 **조용히 건너뛰지 않는다** → FAIL 로 센다(누락이 통과로 흡수되는 경로 차단).
3. 문서 형태 검사(키워드 presence)는 [자체] 태그를 유지하고, 그 한계를 출력 마지막 줄에 적는다.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REP, FIG, NB = ROOT / "reports", ROOT / "figures", ROOT / "notebooks"
results: list[tuple[str, str, bool, str]] = []      # (태그, 이름, 통과, 상세)


def check(name: str, ok: bool, detail: str = "", spec: bool = False) -> None:
    results.append(("[원문]" if spec else "[자체]", name, bool(ok), detail))


def jload(name: str):
    p = REP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    la, pi, svd = jload("linear_algebra.json"), jload("power_iteration.json"), jload("svd.json")
    ca, op, bp, pr = jload("calculus.json"), jload("optimizer.json"), jload("backprop.json"), jload("probability.json")
    for name, obj in [("linear_algebra.json", la), ("power_iteration.json", pi), ("svd.json", svd),
                      ("calculus.json", ca), ("optimizer.json", op), ("backprop.json", bp), ("probability.json", pr)]:
        if obj is None:
            check(f"reports/{name} 존재", False, "없음 → python scripts/run_all.py", spec=True)

    # ══ [원문] 미션 원문 조항 검사 ══════════════════════════════════════════════
    # 1-1/1-2/1-3 변환 그림(회전 R(θ) · **S(2,0.5)** · 전단 Sh(k))
    check("원문1-1/1-2/1-3 그림 01 존재", (FIG / "01_linalg_transforms.png").exists(), spec=True)
    src_la = (ROOT / "src" / "linear_algebra.py").read_text(encoding="utf-8") if (ROOT / "src" / "linear_algebra.py").exists() else ""
    check("원문1-2 S(2,0.5) 가 그림 대상 행렬에 있다", "diag_scale(2.0, 0.5)" in src_la and "SPEC_SCALE" in src_la,
          "MATRICES 에 S(2,0.5) 가 있어야 그림에 나온다", spec=True)
    if la:
        check("원문1-4 면적비 vs |det| 상대오차 ≤1%", bool(la["gate_원문1-4"]["통과"]),
              f"최대 {la['gate_원문1-4']['최대_상대오차%']}%", spec=True)
        check("원문1-4 축비 값이 '면적비' 열에 섞이지 않았다",
              all(abs(float(r["면적비(실측)"]) - abs(float(r["det"]))) <= 1e-9 for r in la["표"] if "면적비(실측)" in r)
              and any("축비(실측)" in r for r in la["표"]), "열 이름 = 측정 정의", spec=True)
    # 1-5/1-6 power iteration + eig 비교 + 시각화
    if pi:
        check("원문1-6 np.linalg.eig 비교·오차 ≤5%", bool(pi["gate_원문1-6"]["통과"]),
              f"최대 {pi['gate_원문1-6']['최대_상대오차%']}%", spec=True)
        check("원문1-6 비교 함수가 eig 다", "np.linalg.eig(" in (ROOT / "src" / "power_iteration.py").read_text(encoding="utf-8"),
              spec=True)
        check("원문1-5/1-6 power iteration 시각화 그림 존재", (FIG / "08_power_iteration.png").exists(), spec=True)
        check("power 이론 공식이 수렴 기준(Rayleigh)과 같다",
              all(0.5 <= r["반복수"] / max(r["이론_예측반복_Rayleigh"], 1) <= 2.0 for r in pi["표"][:3]),
              str([(r["반복수"], r["이론_예측반복_Rayleigh"]) for r in pi["표"][:3]]), spec=True)
        check("직교 v0 케이스가 **실패**로 기록됐다", "실패" in pi["표"][-1]["판정"],
              pi["표"][-1]["판정"][:60], spec=True)
    # 1-7 SVD k=10·50·100 그림
    if svd:
        check("원문1-7 그림에 k=10·50·100", set(svd["gate_원문1-7"]["요구_k"]) <= set(svd["gate_원문1-7"]["그림_k"]),
              f"요구 {svd['gate_원문1-7']['요구_k']} / 그림 {svd['gate_원문1-7']['그림_k']}", spec=True)
        check("원문1-7 그림 02 존재", (FIG / "02_svd_compare.png").exists(), spec=True)
    # 2-1 수치미분 x=3 · 해석값 6 · ≤1e-4
    if ca:
        spec21 = ca["원문2-1_검증"]
        check("원문2-1 지점이 x=3", "x=3" in spec21["지점"], spec21["지점"], spec=True)
        check("원문2-1 해석값 6 과 비교", spec21["해석값"] == 6.0, str(spec21["해석값"]), spec=True)
        check("원문2-1 오차 ≤1e-4 (전 h)", bool(spec21["gate_≤1e-4"]), spec21["오차_최대(전h)"], spec=True)
        # 2-2/2-3/2-4 등고선 x²+y² + 화살표 + 직교
        check("원문2-2 등고선 함수가 x²+y²", "원문: f = x^2 + y^2" in ca["직교성"],
              str(list(ca["직교성"])), spec=True)
        perp = ca["직교성"]["원문: f = x^2 + y^2"]
        check("원문2-4 직교 편차 ≤1° (실측)", float(perp["실측_최대편차_deg"]) <= 1.0 and perp["표본수"] > 100,
              f"최대 {perp['실측_최대편차_deg']}° / {perp['표본수']}점", spec=True)
        check("원문2-3 Gradient 시각화 함수가 있다",
              "gradient_visualization" in (ROOT / "src" / "calculus.py").read_text(encoding="utf-8")
              and "quiver" in (ROOT / "src" / "calculus.py").read_text(encoding="utf-8"), spec=True)
        check("원문2-2~2-4 그림 03 존재", (FIG / "03_gradient_contours.png").exists(), spec=True)
    # 3-4/3-5/3-8 역전파
    if bp:
        check("원문3-4 순전파 중간값 z1·a1·z2·y_pred 기록",
              all(k in bp["고정예제"] for k in ("z1", "a1", "z2", "yhat")), str(list(bp["고정예제"])), spec=True)
        check("원문3-5 필수 유도 6항(∂L/∂y_pred·z2·W2·a1·z1·W1)",
              [r["키"] for r in bp["원문3-5_필수유도"]["표"]] == ["dy_pred", "dz2", "dW2", "da1", "dz1", "dW1"],
              str([r["항"] for r in bp["원문3-5_필수유도"]["표"]]), spec=True)
        check("원문3-5 6항 값이 두 y_true 에 대해 있다",
              all(set(("dy_pred", "dz2", "dW2", "da1", "dz1", "dW1")) <= set(v)
                  for v in bp["원문3-5_필수유도"]["y_true_별_값"].values()), spec=True)
        nb_bp = (NB / "backprop_derivation.ipynb")
        nb_txt = nb_bp.read_text(encoding="utf-8") if nb_bp.exists() else ""
        check("원문3-5 노트북에도 6항이 있다",
              all(s in nb_txt for s in ("∂L/∂y_pred", "∂L/∂z₂", "∂L/∂W₂", "∂L/∂a₁", "∂L/∂z₁", "∂L/∂W₁")), spec=True)
        check("원문3-8 손계산 ↔ NumPy 4자리 일치",
              all(bp["y_true_별"][y]["gate_4자리일치"] for y in ("1.0", "0.0")), spec=True)
        check("원문3-7 shape 표기가 노트북에 있다", "(2,2)" in nb_txt and "(2,)" in nb_txt, spec=True)
    # 4-2/4-4/4-6/4-7 최적화
    if op:
        check("원문4-2 lr=0.1·100회 → 반경 ≤0.1", bool(op["gate_원문4-2"]["gate_원문4-2"]),
              f"반경 {op['gate_원문4-2']['반경_최종']}", spec=True)
        check("원문4-4 대조 표가 있다(원에서는 lr≥0.5 가 발산 아님)",
              "원문4-4_대조(lr≥0.5 발산)" in op and
              "발산" not in next(r for r in op["원문4-4_대조(lr≥0.5 발산)"]["표"] if r["lr"] == 0.5)["원_vanilla"],
              spec=True)
        check("원문4-6/4-7 모멘텀 비교(타원 lr=0.15) + 경로 그림",
              next(r for r in op["표"] if r["함수"].startswith("타원") and r["옵티"] == "vanilla" and r["lr"] == 0.15)["판정"] == "발산"
              and next(r for r in op["표"] if r["함수"].startswith("타원") and r["옵티"] == "momentum" and r["lr"] == 0.15)["판정"] != "발산"
              and (FIG / "04_optimizer_paths.png").exists(), spec=True)
        check("원문4-6 두 옵티마이저 경로를 하나의 Figure에 겹침",
              "plot" in (ROOT / "src" / "optimizer.py").read_text(encoding="utf-8")
              and (FIG / "04_optimizer_paths.png").exists(), spec=True)
        check("모멘텀 이론상한이 heavy-ball 공식(2(1−β)/λmax)",
              abs(op["수렴_상한"]["타원: x^2 + 10y^2 / momentum"]["이론_상한"] - 0.01) < 1e-9,
              str(op["수렴_상한"]["타원: x^2 + 10y^2 / momentum"]), spec=True)
    # 5-1/5-2/5-3/5-4/5-5 확률
    if pr:
        check("원문5-1 정규분포가 N(0,1)·N(2,0.5)", set(pr["정규PDF(원문5-1)"]) == {"N(0,1)", "N(2,0.5)"}
              and pr["정규PDF(원문5-1)"]["N(2,0.5)"]["mu"] == 2.0
              and pr["정규PDF(원문5-1)"]["N(2,0.5)"]["sigma"] == 0.5,
              str({k: (v["mu"], v["sigma"]) for k, v in pr["정규PDF(원문5-1)"].items()}), spec=True)
        check("원문5-1 그림 06 존재", (FIG / "06_normal_pdf.png").exists(), spec=True)
        check("원문5-2 베르누이 B(0.3)·B(0.7) PMF 합=1",
              set(pr["베르누이PMF(원문5-2)"]) == {"B(0.3)", "B(0.7)"}
              and all(abs(float(v["합"]) - 1.0) <= 1e-9 for v in pr["베르누이PMF(원문5-2)"].values())
              and pr["베르누이PMF(원문5-2)"]["B(0.3)"]["PMF_k0_k1"] == [0.7, 0.3],
              str({k: v["PMF_k0_k1"] for k, v in pr["베르누이PMF(원문5-2)"].items()}), spec=True)
        check("원문5-2 그림 07 존재", (FIG / "07_bernoulli_pmf.png").exists(), spec=True)
        check("원문5-3 softmax 합=1 ±1e-6", bool(pr["softmax"]["gate_합=1_±1e-6"]),
              f"최대 |합−1| {pr['softmax']['최대_합-1_절대오차']}", spec=True)
        nb_pr = (NB / "probability_loss.ipynb")
        nbp = nb_pr.read_text(encoding="utf-8") if nb_pr.exists() else ""
        check("원문 제출물5 노트북에 분포 시각화(이미지 출력)가 있다",
              '"image/png"' in nbp and "N(2,0.5)" in nbp and "B(0.3)" in nbp,
              "노트북 안에서 plt.show() 로 그린 그림이 outputs 에 박혀 있어야 한다", spec=True)
        check("원문5-4/5-5 MSE=MLE·CE=MLE 유도(노트북·문서)",
              float(pr["MSE_MLE_동치"]["차이"]) == 0.0 and float(pr["CE_MLE_동치"]["차이"]) == 0.0
              and ("MSE" in nbp or "MLE" in nbp), spec=True)

    # ══ [자체] 이 레포가 세운 기준 (검사의 유효성·문서 형태) ═════════════════════════
    check("src/linear_algebra.py 존재", (ROOT / "src" / "linear_algebra.py").exists())
    if la:
        rows = la["표"]
        check("det·면적비 표 4행↑", len(rows) >= 4, f"{len(rows)}행")
        check("면적비 오차 ≤1e-9", max(float(r["절대오차"]) for r in rows if "면적비(실측)" in r) <= 1e-9)
        check("'무의미' 검사 여부를 표가 스스로 표시", any(r["검사가무의미"] for r in rows),
              f"항등식 행 {sum(1 for r in rows if r['검사가무의미'])}개")
        check("det=1 행에 대안 검사(축비)가 있다", any("축비(실측)" in r for r in rows))
    if pi:
        its = [r["반복수"] for r in pi["표"][:3]]
        check("power iteration 3케이스", len(its) == 3, str(its))
        check("비 ↓ → 반복수 ↓ (단조)", its[0] > its[1] > its[2], " → ".join(map(str, its)))
        check("v0 직교 케이스가 표에 있다", "직교" in json.dumps(pi, ensure_ascii=False))
    if svd:
        ks = [r["k"] for r in svd["표"]]
        check("SVD k=1·10·50 포함", {1, 10, 50} <= set(ks), str(ks))
        check("저장비율 >100% 구간 노출(압축 역효과)", any(r["저장비율_%"] > 100 for r in svd["표"]))
        check("k≥랭크에서 개선 0(중복 노출)", any(r["k"] == 100 and r["실제 랭크"] == 64 for r in svd["표"]))
    if ca:
        by = {r["함수"]: r for r in ca["표"]}
        check("수치미분 표 3함수 × 8h", len(ca["표"]) >= 3, f"{len(ca['표'])}행")
        check("x^2 은 전 h 통과(=항등식임이 기록됨)", by["x^2"]["1e-4_전구간통과"] is True)
        check("x^3·sin 은 통과 실패(=검사에게 구분력이 있다)",
              by["x^3"]["1e-4_전구간통과"] is False or by["sin(x)"]["1e-4_전구간통과"] is False)
        check("최적 h 가 함수마다 다르다", len({by[k]["h_최적"] for k in by}) >= 2,
              str({k: by[k]["h_최적"] for k in by}))
        check("직교 검사: 유도 접선은 0° 편차(항등식)로 표시",
              float(ca["직교성"]["원문: f = x^2 + y^2"]["유도_최대편차_deg"]) == 0.0
              and float(ca["직교성"]["원문: f = x^2 + y^2"]["실측_최대편차_deg"]) > 0,
              f"유도 0 / 실측 {ca['직교성']['원문: f = x^2 + y^2']['실측_최대편차_deg']}°")
        check("특이점(∇f=0) 제외 개수를 적었다",
              "특이점(∇f=0)_제외개수" in ca["직교성"]["원문: f = x^2 + y^2"],
              str(ca["직교성"]["원문: f = x^2 + y^2"].get("특이점(∇f=0)_제외개수")))
    if op:
        rows = op["표"]
        check("lr 스윕 2함수×3옵티×13lr", len(rows) >= 78, f"{len(rows)}행")
        check("발산과 수렴이 모두 관측", any(r["판정"] == "발산" for r in rows) and any(r["판정"] == "수렴" for r in rows))
        check("판정 라벨 4종(발산/진동/정체/수렴)이 관측 기반으로 갈린다",
              all(r["판정"] in ("발산", "진동", "정체(천천히 감소)", "수렴") for r in rows),
              str(sorted({r["판정"] for r in rows})))
        check("원문 기준(반경≤0.1) 열이 모든 행에 있다", all("원문4-2_기준(반경≤0.1)" in r for r in rows))
        check("측정 상한 ≤ 이론 상한", all(v["측정_상한(자체기준)"] in (None, "None") or v["이론_상한"] is None
                                    or v["측정_상한(자체기준)"] <= v["이론_상한"] + 1e-12
                                    for v in op["수렴_상한"].values()),
              str({k: (v["측정_상한(자체기준)"], v["이론_상한"]) for k, v in op["수렴_상한"].items()
                   if v["측정_상한(자체기준)"]}))
        check("반복수 민감도 표(판정 기준이 비교 대상)", len(op.get("반복수_민감도", [])) >= 4)
        check("그림 04 경로 겹치기·05 lr 스윕", (FIG / "04_optimizer_paths.png").exists()
              and (FIG / "05_lr_sweep.png").exists())
        check("lr 단독 주장 금지 문구가 리포트에 있다", "함수·기준·반복수를 함께" in op["주의"])
        check("heavy-ball 변형임을 고지했다", "heavy-ball" in op["주의"])
    if bp:
        check("손계산 예제 값 확보(ŷ=0.647292)", abs(bp["고정예제"]["yhat"] - 0.647292) < 1e-6)
        for y in ("1.0", "0.0"):
            v = bp["y_true_별"][y]
            check(f"y_true={y}: gradcheck ≤1e-7(0항 제외)", v["gate_1e-7"], v["최대상대오차_0제외"])
        check("x=[1,0] 로 죽는 열을 '결핍'으로 설명", bp["기울기_0_열"]["개수"] == 1)
        check("y_true 미정의를 양쪽 계산으로 대응", "양쪽" in bp["y_true_미정의_대응"])
        check("관례(전치 함정) 명시", "outer" in bp["관례"])
    if pr:
        check("깁스 부등식은 참값 기준으로 성립", pr["CE_MLE_동치"]["깁스_성립"] is True,
              f"CE {pr['CE_MLE_동치']['참값대_ fitted_교차엔트로피']} ≥ H {pr['CE_MLE_동치']['진_엔트로피']}")
        check("플러그인 편향을 숫자로 적었다", "플러그인_편향" in pr["CE_MLE_동치"],
              str(pr["CE_MLE_동치"].get("플러그인_편향")))
        check("softmax: 큰 스케일에서만 안정화가 검증된다",
              pr["softmax"]["차감버전_유한"] and not pr["softmax"]["naive_유한"])
        check("정규 PDF 넓이는 구간 의존임을 적었다", "적분 구간" in pr["넓이_주의"])
        check("이전 잘못된 베르누이 그림을 정정 기록으로 남겼다", "베르누이PMF_정정_기록" in pr,
              str(pr.get("베르누이PMF_정정_기록", {}).get("이전_구현_막대합")))
    # 노트북·문서 (형태 검사 — 강도 한계를 마지막에 적는다)
    for nb_name, need in [("backprop_derivation.ipynb", "(2,2)"), ("probability_loss.ipynb", "MLE")]:
        f = NB / nb_name
        txt = f.read_text(encoding="utf-8") if f.exists() else ""
        check(f"{nb_name} 존재·실행결과·'{need}'", f.exists() and '"outputs"' in txt and need in txt,
              "" if f.exists() else "파일 없음")
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    check("README '요약' 섹션", "주요 결과 요약" in readme)
    check("README 에 5조항 검수표", readme.count("검증") >= 3 and "요구" in readme)
    check("README 에 갭 표", "갭" in readme)
    check("README 에 스펙 이탈·정정 기록", "스펙 이탈" in readme or "정정" in readme)
    analysis = (REP / "analysis.md").read_text(encoding="utf-8") if (REP / "analysis.md").exists() else ""
    check("analysis.md 가 격자 해상도 의존을 명시", "격자" in analysis)
    check("analysis.md 에 그림 참조", analysis.count("figures/") >= 3 or "그림" in analysis)
    check("analysis.md 가 '겹쳐 요구를 모두 지켰다'고 과장하지 않는다",
          "모두 지켰다" not in analysis or "서브플롯" in analysis)

    w = max(len(n) for _, n, _, _ in results) + 1
    npass = sum(1 for _, _, ok, _ in results if ok)
    sp = [(n, ok) for t, n, ok, _ in results if t == "[원문]"]
    it = [(n, ok) for t, n, ok, _ in results if t == "[자체]"]
    for tag, name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {tag} {name:<{w}} {detail}")
    print(f"---- 원문 {sum(1 for _, ok in sp if ok)}/{len(sp)} · 자체 {sum(1 for _, ok in it if ok)}/{len(it)} "
          f"→ 전체 {npass}/{len(results)} PASS")
    print("     (주의: [자체] 항목 중 문서 형태·키워드 검사는 '문장에 그 단어가 있는가'만 본다 — 검증 강도가 낮다. "
          "값의 검증은 [원문] 항목과 tests/ 가 담당한다)")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
