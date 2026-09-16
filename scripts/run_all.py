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
    figures = sorted(p.name for p in (ROOT / "figures").glob("*.png"))
    metrics = {"런": "A2-1 (AI 학습의 수학)", "산출_그림": figures,
               "요약": {
                   "det_면적비_최대오차": max(float(r["절대오차"]) for r in parts["linear_algebra"]["표"]),
                   "power_반복수": [r["반복수"] for r in parts["power_iteration"]["표"][:3]],
                   "svd_k50_설명분산": [r for r in parts["svd"]["표"] if r["k"] == 50][0]["설명분산_%"],
                   "svd_k50_저장비율": [r for r in parts["svd"]["표"] if r["k"] == 50][0]["저장비율_%"],
                   "수치미분_x2_전구간_1e-4통과": parts["calculus"]["표"][0]["1e-4_전구간통과"],
                   "등고선_직교_실측최대편차_deg": parts["calculus"]["직교성"]["실측_최대편차_deg"],
                   "타원_lr0.15_vanilla": parts["optimizer"]["핵심_실측"]["타원_vanilla_lr0.15"],
                   "타원_lr0.15_momentum": parts["optimizer"]["핵심_실측"]["타원_momentum_lr0.15"],
                   "backprop_손계산": parts["backprop"]["y_true_별"]["1.0"]["dW1"],
                   "backprop_최대상대오차": parts["backprop"]["y_true_별"]["1.0"]["최대상대오차_0제외"],
                   "MSE_MLE_차이": parts["probability"]["MSE_MLE_동치"]["차이"],
                   "CE_MLE_차이": parts["probability"]["CE_MLE_동치"]["차이"],
               },
               "각_파일": {k: f"{k}.json" for k in parts}}
    (REP / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = ["# A2-1 분석 리포트 (이 파일은 run_all.py 가 생성한다)", "",
          "모든 숫자는 `reports/*.json`(같은 실행의 산출물) 에서 복사됐다. 손으로 고치면 다음 실행에서 사라진다.", "",
          "## 1. 선형대수: det = 면적배 인가", "",
          "| 행렬 | det | 면적비 | 절대오차 | 이 검사 무의미? |", "|---|---|---|---|---|"]
    for r in parts["linear_algebra"]["표"]:
        md.append(f"| {r['행렬']} | {r['det']} | {r['면적비(실측)']} | {r['절대오차']} | "
                  f"{'**예** (det=1 → 항등식)' if r['검사가무의미'] else '아니오'} |")
    md += ["", "그림: `figures/01_linalg_transforms.png` — 회전은 원의 모양을 보존하고(축 길이 1), 전단은 타원을 기울이게 한다. "
               "넓이는 세 경우 모두 보존되거나(det=1) 1.5배가 된다(det=1.5).", "",
           f"> {parts['linear_algebra']['판정']}", "",
           "## 2. power iteration: 수렴 속도는 λ2/λ1", "",
           "| 케이스 | λ 실측 | 상대오차% | 반복수 | 이론 예측 |", "|---|---|---|---|---|"]
    for r in parts["power_iteration"]["표"]:
        md.append(f"| {r['case']} | {r['λ_실측']} | {r['상대오차%']} | {r['반복수']} | {r['이론_예측반복']} |")
    md += ["", f"> {parts['power_iteration']['판정']}", "",
           "## 3. SVD 압축: k=50 은 압축이 아니다", "",
           "| k | mse | PSNR dB | 저장비율% | 설명분산% |", "|---|---|---|---|---|"]
    for r in parts["svd"]["표"]:
        md.append(f"| {r['k']} | {r['mse']} | {r['psnr_dB']} | {r['저장비율_%']} | {r['설명분산_%']} |")
    md += ["", "그림: `figures/02_svd_compare.png` — k=1 에서 이미 사인파의 방향이 살고, k=50 에서 눈으로는 구별이 어려운데 "
               "저장량은 원본의 157.5% 로 **늘어난다**(랭크 64 상한).", "",
           f"> {parts['svd']['판정']}", "",
           "## 4. 수치미분: x^2 에서 ≤1e-4 검사는 항등식", "",
           "| 함수 | 최적 h | 최적 오차 | h=1e-1 | h=1e-8 | 1e-4 전구간 통과 |", "|---|---|---|---|---|---|"]
    for r in parts["calculus"]["표"]:
        md.append(f"| {r['함수']} | {r['h_최적']} | {r['오차_최적']} | {r['h=1e-1']} | {r['h=1e-8']} | "
                  f"{'**O**' if r['1e-4_전구간통과'] else 'X'} |")
    md += ["", f"> {parts['calculus']['판정']}", "",
           f"등고선-그라디언트 직교: 그려진 곡선 기준 최대 편차 **{parts['calculus']['직교성']['실측_최대편차_deg']}°**, "
           f"접선을 그라디언트로 유도하면 **{parts['calculus']['직교성']['유도_최대편차_deg']}°**(항등식). "
           f"{parts['calculus']['직교성']['해석']}", "",
           "그림: `figures/03_gradient_contours.png`(등고선과 그라디언트 화살표), "
           "`figures/04_optimizer_paths.png`(두 옵티마이저 경로를 같은 축에), `figures/05_lr_sweep.png`(함수별 상한) — "
           "미션이 요구한 '하나의 Figure에 겹쳐'를 모두 지켰다.", "",
           "## 5. 최적화: lr 상한은 조건수에서 나온다", "",
           "| 케이스 | 측정 상한 | 이론 2/λmax |", "|---|---|---|"]
    for k, v in parts["optimizer"]["수렴_상한"].items():
        md.append(f"| {k} | {v['측정_상한']} | {v['이론_상한']} |")
    md += ["", f"- 타원 lr=0.15: vanilla **{parts['optimizer']['핵심_실측']['타원_vanilla_lr0.15']}** vs "
               f"momentum **{parts['optimizer']['핵심_실측']['타원_momentum_lr0.15']}** (100회 시점)",
           f"- {parts['optimizer']['주의']}", "",
           "반복수를 늘리면 판정이 바뀌는 구간이 있다(아래는 momentum 이 1000회에서 수렴):", "",
           "| 함수 | lr | vanilla 100 | momentum 100 | vanilla 1000 | momentum 1000 |", "|---|---|---|---|---|---|"]
    for r in parts["optimizer"]["반복수_민감도"]:
        md.append(f"| {r['함수']} | {r['lr']} | {r['van_100']} | {r['mom_100']} | {r['van_1000']} | {r['mom_1000']} |")
    md += ["", "## 6. 역전파: 손계산 = NumPy = 수치미분 (4자리)", "",
           f"- 관례: {parts['backprop']['관례']}", "- 고정 예제: " + json.dumps(parts['backprop']['고정예제'], ensure_ascii=False), ""]
    for y, v in parts["backprop"]["y_true_별"].items():
        md.append(f"- **y_true={y}**: L={v['L']}, ŷ={v['yhat']}, ∂L/∂z₂={v['dz2']}, "
                  f"∂L/∂W₁={v['dW1']}, 최대상대오차(0 제외)={v['최대상대오차_0제외']}, "
                  f"gate ≤1e-7: {v['gate_1e-7']} / 4자리 일치: {v['gate_4자리일치']}")
    md += ["", f"- {parts['backprop']['y_true_미정의_대응']}", f"- {parts['backprop']['기울기_0_열']['해석']}"
                  f" (0인 열 {parts['backprop']['기울기_0_열']['개수']}개)", "",
           "## 7. 손실 = 최대우도", "",
           f"- MSE=MLE: 기울기 {parts['probability']['MSE_MLE_동치']['기울기_MSE최소']} vs "
           f"{parts['probability']['MSE_MLE_동치']['기울기_로그우도최대']} (차이 {parts['probability']['MSE_MLE_동치']['차이']})",
           f"- CE=MLE: p̂ {parts['probability']['CE_MLE_동치']['로그우도_최대의_p']} vs "
           f"{parts['probability']['CE_MLE_동치']['BCE_최소의_p']} / 깁스 부등식은 참값 기준 성립"
           f"({parts['probability']['CE_MLE_동치']['깁스_성립']}), 표본 플러그인은 {parts['probability']['CE_MLE_동치']['플러그인_편향']} 아래로 왜곡",
           f"- softmax: {parts['probability']['softmax']['해석']}", ""]
    (REP / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(metrics["요약"], ensure_ascii=False, indent=1))
    print(f"그림 {len(figures)}장 · 리포트 {len((REP / 'analysis.md').read_text(encoding='utf-8').splitlines())}줄")


if __name__ == "__main__":
    main()
