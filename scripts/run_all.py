"""A2-1 전체 재현 파이프라인: src 모듈을 순서대로 실행하고 reports/metrics.json · reports/analysis.md 를 쓴다.

실행: `python scripts/run_all.py`  (data 없음 · 결정론적 · 시드는 probability 안에서만 사용)
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import backprop  # noqa: E402
import calculus  # noqa: E402
import linear_algebra  # noqa: E402
import optimizer  # noqa: E402
import power_iteration  # noqa: E402
import probability  # noqa: E402
import svd_compress  # noqa: E402

REP = ROOT / "reports"


def main() -> None:
    parts = {
        "linear_algebra": linear_algebra.main(),
        "power_iteration": power_iteration.main(),
        "svd": svd_compress.main(),
        "calculus": calculus.main(),
        "optimizer": optimizer.main(),
        "backprop": backprop.main(),
        "probability": probability.main(),
    }
    la, pi, sv, ca, op, bp, pr = (parts["linear_algebra"], parts["power_iteration"], parts["svd"],
                                  parts["calculus"], parts["optimizer"], parts["backprop"], parts["probability"])
    figures = sorted(p.name for p in (ROOT / "figures").glob("*.png"))
    metrics = {"런": "A2-1 (AI 학습의 수학)", "산출_그림": figures,
               "요약": {
                   "det_면적비_최대상대오차%": la["gate_원문1-4"]["최대_상대오차%"],
                   "gate_원문1-4(면적비≤1%)": la["gate_원문1-4"]["통과"],
                   "power_반복수": [r["반복수"] for r in pi["표"][:3]],
                   "power_이론예측_Rayleigh": [r["이론_예측반복_Rayleigh"] for r in pi["표"][:3]],
                   "gate_원문1-6(eig비교≤5%)": pi["gate_원문1-6"]["통과"],
                   "svd_k50_설명분산": [r for r in sv["표"] if r["k"] == 50][0]["설명분산_%"],
                   "svd_k50_저장비율": [r for r in sv["표"] if r["k"] == 50][0]["저장비율_%"],
                   "gate_원문1-7(그림k)": sv["gate_원문1-7"]["그림_k"],
                   "gate_원문2-1(x=3,≤1e-4)": ca["원문2-1_검증"]["gate_≤1e-4"],
                   "수치미분_x2_오차최대(전h)": [r for r in ca["표"] if r["함수"] == "x^2"][0]["오차_최대(전h)"],
                   "등고선_직교_실측최대편차_deg(원문함수)": ca["직교성"]["원문: f = x^2 + y^2"]["실측_최대편차_deg"],
                   "gate_원문4-2(반경≤0.1)": op["gate_원문4-2"]["gate_원문4-2"],
                   "타원_lr0.15_vanilla": op["핵심_실측"]["타원_vanilla_lr0.15"],
                   "타원_lr0.15_momentum": op["핵심_실측"]["타원_momentum_lr0.15"],
                   "backprop_손계산": bp["y_true_별"]["1.0"]["dW1"],
                   "backprop_최대상대오차": bp["y_true_별"]["1.0"]["최대상대오차_0제외"],
                   "MSE_MLE_차이": pr["MSE_MLE_동치"]["차이"],
                   "CE_MLE_차이": pr["CE_MLE_동치"]["차이"],
                   "gate_원문5-3(softmax합=1±1e-6)": pr["softmax"]["gate_합=1_±1e-6"],
                   "베르누이PMF_합": {k: v["합"] for k, v in pr["베르누이PMF(원문5-2)"].items()},
               },
               "각_파일": {k: f"{k}.json" for k in parts}}
    (REP / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = ["# A2-1 분석 리포트 (이 파일은 run_all.py 가 생성한다)", "",
          "모든 숫자는 `reports/*.json`(같은 실행의 산출물) 에서 복사됐다. 손으로 고치면 다음 실행에서 사라진다.",
          "각 절의 `gate_원문…` 은 **미션 원문 조항**을 직접 판정한 것이고, 그 외 검사는 이 레포가 세운 자체 기준이다.", "",
          "## 1. 선형대수: det = 면적배 인가", "",
          "| 행렬 | det | 면적비(실측) | 축비(실측) | 절대오차 | 상대오차% | 원문 1-4(≤1%) | 이 검사 무의미? |",
          "|---|---|---|---|---|---|---|---|"]
    for r in la["표"]:
        md.append(f"| {r['행렬']} | {r['det']} | {r['면적비(실측)']} | {r.get('축비(실측)', '—')} | {r['절대오차']} | "
                  f"{r['상대오차%']} | {r['gate_원문1-4(≤1%)']} | "
                  f"{'**예** (det=1 → 항등식)' if r['검사가무의미'] else '아니오'} |")
    md += ["", f"그림: `figures/01_linalg_transforms.png` — 원문 1-1(회전)·1-2(**S(2,0.5)**)·1-3(전단) 을 "
               f"하나의 Figure(서브플롯 4)에 싣고, 각 축에서 변환 전후를 겹쳤다. 마지막 축은 det≠1 대조(스케일 3,0.5). "
               f"S(2,0.5) 는 det=1 이라 면적비가 1.0 으로 '통과'하지만 **정보가 없다** → 축비 {la['표'][-1].get('축비(실측)')} 로 "
               "모양 변화를 따로 잰다(축비는 별도 열에 둔다 — 면적비 열에 섞지 않는다).", "",
           f"> {la['판정']}", "",
           f"> 원문 1-4 게이트: {json.dumps(la['gate_원문1-4'], ensure_ascii=False)}", "",
           "## 2. power iteration: 수렴 속도는 λ2/λ1", "",
           "| 케이스 | λ 실측 | λ 참값(eig) | 상대오차%(vs eig) | 원문 1-6(≤5%) | 반복수 | 이론 예측(Rayleigh) | 판정 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in pi["표"]:
        md.append(f"| {r['case']} | {r['λ_실측']} | {r['λ_참값_eig']} | {r['상대오차%(vs eig)']} | "
                  f"{r['gate_원문1-6(≤5%)']} | {r['반복수']} | {r['이론_예측반복_Rayleigh']} | {r['판정'][:60]} |")
    md += ["", f"그림: `figures/08_power_iteration.png` — Rayleigh quotient 상대오차의 수렴 곡선(r=0.9/0.5/0.1)과 "
               "실측 정지점·이론 예측 횟수(원문 1-5/1-6 의 시각화 요구).", "",
           f"> {pi['판정']}", "",
           f"> 원문 1-6 게이트: {json.dumps(pi['gate_원문1-6'], ensure_ascii=False)}", "",
           "## 3. SVD 압축: k=50 은 압축이 아니다", "",
           "| k | 실제 랭크 | mse | PSNR dB | 저장비율% | 설명분산% | 비고 |", "|---|---|---|---|---|---|---|"]
    for r in sv["표"]:
        md.append(f"| {r['k']} | {r['실제 랭크']} | {r['mse']} | {r['psnr_dB']} | {r['저장비율_%']} | "
                  f"{r['설명분산_%']} | {r['k_요청_대비_랭크_상한']} |")
    md += ["", f"그림: `figures/02_svd_compare.png` — 원문 1-7 이 요구한 **k=10·50·100**(+ 대조 k=1) 을 하나의 Figure에. "
               f"k=100 은 64×64 의 랭크 상한(64) 때문에 k=64 와 같은 그림이 되며 제목에 적는다. "
               f"k=50 은 눈으로 구별이 어려운데 저장량은 원본의 "
               f"{[r for r in sv['표'] if r['k'] == 50][0]['저장비율_%']}% 로 **늘어난다**.", "",
           f"> {sv['판정']}", "",
           f"> 원문 1-7 게이트: {json.dumps(sv['gate_원문1-7'], ensure_ascii=False)}", "",
           "## 4. 수치미분: x^2 에서 ≤1e-4 검사는 항등식", "",
           f"원문 2-1 조건: f(x)=x², **{ca['원문2-1_검증']['지점']}**, 해석값 {ca['원문2-1_검증']['해석값']} 과 비교 → "
           f"오차 최대 {ca['원문2-1_검증']['오차_최대(전h)']} (≤1e-4: **{ca['원문2-1_검증']['gate_≤1e-4']}**)", "",
           "| 함수 | x | 해석값 | 최적 h | 최적 오차 | h=1e-1 | h=1e-8 | 오차 최대(전 h) | 1e-4 전구간 통과 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in ca["표"]:
        md.append(f"| {r['함수']} | {r['x']:g} | {r['해석값']} | {r['h_최적']} | {r['오차_최적']} | {r['h=1e-1']} | "
                  f"{r['h=1e-8']} | {r['오차_최대(전h)']} | {'**O**' if r['1e-4_전구간통과'] else 'X'} |")
    md += ["", "지점 민감도 대조(x=1.0, 이전 구현이 쓰던 지점): "
               f"{json.dumps(ca['민감도_지점(x=1.0)'], ensure_ascii=False)}", "",
           f"> {ca['판정']}", ""]
    perp_spec = ca["직교성"]["원문: f = x^2 + y^2"]
    perp_extra = ca["직교성"]["추가(타원): f = x^2 + 4y^2"]
    md += [f"등고선-그라디언트 직교(원문 2-2 함수 **x²+y²**): 그려진 곡선 기준 최대 편차 "
           f"**{perp_spec['실측_최대편차_deg']}°**(중위 {perp_spec['실측_중위편차_deg']}°, {perp_spec['표본수']}점, "
           f"∇f=0 특이점 {perp_spec['특이점(∇f=0)_제외개수']}개 제외), 접선을 그라디언트로 유도하면 "
           f"**{perp_spec['유도_최대편차_deg']}°**(항등식). 타원 x²+4y² 대조: 최대 {perp_extra['실측_최대편차_deg']}°. "
           f"{ca['직교성']['해석']}", "",
           f"그라디언트 시각화 함수(원문 제출물 2): {json.dumps(ca['그라디언트_시각화'], ensure_ascii=False)}", "",
           "그림: `figures/03_gradient_contours.png`(등고선+그라디언트 화살표 — 원문 함수와 타원 대조를 "
           "**하나의 Figure**에 서브플롯 2개로), `figures/04_optimizer_paths.png`(두 옵티마이저 경로를 같은 축에), "
           "`figures/05_lr_sweep.png`(함수별 상한 점선 2종). '하나의 Figure에 겹쳐' 요구 4건 중 "
           "**01(축 4)·02(축 4)·03(축 2)·05(축 2)는 서브플롯**이고, 같은 축에 진짜로 겹친 것은 04(경로)와 06(PDF)·08(곡선)이다 "
           "→ 이 차이는 README 갭 표에도 적는다.", "",
           "## 5. 최적화: lr 상한은 조건수에서 나온다", "",
           "| 케이스 | 측정 상한(자체 f<1e-8) | 측정 상한(원문 반경≤0.1) | 이론 상한 | 공식 |", "|---|---|---|---|---|"]
    for k, v in op["수렴_상한"].items():
        md.append(f"| {k} | {v['측정_상한(자체기준)']} | {v['측정_상한(원문기준)']} | {v['이론_상한']} | {v['이론_공식']} |")
    md += ["", f"- 원문 4-2 게이트: {json.dumps(op['gate_원문4-2'], ensure_ascii=False)}",
           f"- 타원 lr=0.15: vanilla **{op['핵심_실측']['타원_vanilla_lr0.15']}** vs "
           f"momentum **{op['핵심_실측']['타원_momentum_lr0.15']}** (100회 시점)",
           f"- {op['주의']}", "",
           "### 원문 4-4 와 실측이 어긋나는 지점", "",
           "| lr | 원 vanilla | 원 momentum | 타원 vanilla | 타원 momentum |", "|---|---|---|---|---|"]
    for r in op["원문4-4_대조(lr≥0.5 발산)"]["표"]:
        md.append(f"| {r['lr']} | {r['원_vanilla']} | {r['원_momentum']} | {r['타원_vanilla']} | {r['타원_momentum']} |")
    md += ["", f"> {op['원문4-4_대조(lr≥0.5 발산)']['판정']}", "",
           "반복수를 늘리면 판정이 바뀌는 구간이 있다(아래는 momentum 이 1000회에서 수렴):", "",
           "| 함수 | lr | vanilla 100 | momentum 100 | vanilla 1000 | momentum 1000 |", "|---|---|---|---|---|---|"]
    for r in op["반복수_민감도"]:
        md.append(f"| {r['함수']} | {r['lr']} | {r['van_100']} | {r['mom_100']} | {r['van_1000']} | {r['mom_1000']} |")
    md += ["", "## 6. 역전파: 손계산 = NumPy = 수치미분 (4자리)", "",
           f"- 관례: {bp['관례']}",
           "- 원문 3-5 필수 유도 6항: " + json.dumps([r["항"] for r in bp["원문3-5_필수유도"]["표"]], ensure_ascii=False)
           + f" → {bp['원문3-5_필수유도']['판정']}",
           "- 고정 예제: " + json.dumps(bp["고정예제"], ensure_ascii=False), ""]
    for y, v in bp["원문3-5_필수유도"]["y_true_별_값"].items():
        md.append(f"- **y_true={y}** 6항: " + json.dumps(v, ensure_ascii=False))
    md += [""]
    for y, v in bp["y_true_별"].items():
        md.append(f"- **y_true={y}**: L={v['L']}, ŷ={v['yhat']}, ∂L/∂y_pred={v['dy_pred(∂L/∂y_pred)']}, "
                  f"∂L/∂z₂={v['dz2']}, ∂L/∂a₁={v['da1(∂L/∂a₁)']}, ∂L/∂b₁={v['db1']}, ∂L/∂b₂={v['db2']}, "
                  f"∂L/∂W₁={v['dW1']}, 최대상대오차(0 제외)={v['최대상대오차_0제외']}, "
                  f"gate ≤1e-7: {v['gate_1e-7']} / 4자리 일치: {v['gate_4자리일치']}")
    md += ["", f"- {bp['y_true_미정의_대응']}", f"- {bp['기울기_0_열']['해석']}"
                  f" (0인 열 {bp['기울기_0_열']['개수']}개)", "",
           "## 7. 손실 = 최대우도", "",
           "| 분포(원문 지정) | peak | ∫[-8,8] | ∫[-40,40] |", "|---|---|---|---|"]
    for label, v in pr["정규PDF(원문5-1)"].items():
        md.append(f"| {label} (μ={v['mu']}, σ={v['sigma']}) | {v['peak']} | {v['적분[-8,8]']} | {v['적분[-40,40]']} |")
    md += ["", "| 베르누이(원문 지정) | PMF k=0,1 | 합 | 합−1 |", "|---|---|---|---|"]
    for label, v in pr["베르누이PMF(원문5-2)"].items():
        md.append(f"| {label} | {v['PMF_k0_k1']} | {v['합']} | {v['합-1_절대오차']} |")
    md += ["", f"> 이전 구현 정정 기록: {pr['베르누이PMF_정정_기록']['이전_구현']} → "
               f"막대 합 {json.dumps(pr['베르누이PMF_정정_기록']['이전_구현_막대합'], ensure_ascii=False)} (1 이 아님). "
               f"{pr['베르누이PMF_정정_기록']['왜_틀렸나']}", "",
           f"- softmax 원문 5-3(합=1 ±1e-6): **{pr['softmax']['gate_합=1_±1e-6']}** "
           f"(최대 |합−1| {pr['softmax']['최대_합-1_절대오차']}) — "
           + json.dumps({k: v["합-1_절대오차"] for k, v in pr["softmax"]["합=1_검증(원문5-3)"].items()}, ensure_ascii=False),
           f"- MSE=MLE: 기울기 {pr['MSE_MLE_동치']['기울기_MSE최소']} vs "
           f"{pr['MSE_MLE_동치']['기울기_로그우도최대']} (차이 {pr['MSE_MLE_동치']['차이']})",
           f"- CE=MLE: p̂ {pr['CE_MLE_동치']['로그우도_최대의_p']} vs "
           f"{pr['CE_MLE_동치']['BCE_최소의_p']} / 깁스 부등식은 참값 기준 성립"
           f"({pr['CE_MLE_동치']['깁스_성립']}), 표본 플러그인은 {pr['CE_MLE_동치']['플러그인_편향']} 아래로 왜곡",
           f"- softmax 안정성: {pr['softmax']['해석']}", ""]
    (REP / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(metrics["요약"], ensure_ascii=False, indent=1))
    print(f"그림 {len(figures)}장 · 리포트 {len((REP / 'analysis.md').read_text(encoding='utf-8').splitlines())}줄")


if __name__ == "__main__":
    main()
