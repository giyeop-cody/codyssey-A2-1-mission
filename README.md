<!-- hand-authored -->
# A2-1 — AI 학습을 수학으로 직접 풀어보기 (제출물)

- 학습·계획·결정 기록: [codyssey-A2-1-studylog](https://github.com/giyeop-cody/codyssey-A2-1-studylog)
- 이 미션의 일정·게이트: [`docs/learning-map.md`](docs/learning-map.md) · 확률 유도: [`docs/probability_loss.md`](docs/probability_loss.md) · 예제 정의: [`docs/y_true-definition.md`](docs/y_true-definition.md)
- 과정 전체: [codyssey-A-studylog-hub](https://github.com/giyeop-cody/codyssey-A-studylog-hub)
- 이 레포에 적힌 모든 숫자는 `reports/*.json` 에 있고, 그 파일은 `scripts/run_all.py` 가 만든다(손으로 채운 숫자는 없다).

## 실행

```bash
pip install -r requirements.txt          # numpy·matplotlib·pytest·nbformat·nbclient
python scripts/run_all.py                # src 5종을 돌려 reports/*.json·analysis.md·그림 7장 생성
python scripts/build_notebook.py         # 노트북 2종을 **실행해서** outputs 와 함께 저장
python scripts/record_gates.py           # 게이트 판정 → reports/gates.json
python scripts/check_requirements.py     # 요구 5조항 자가채점 (미달 시 exit 1)
python -m pytest -q                       # 기본
MISSION_STRICT=1 python -m pytest -q       # 제출 직전: skip 도 실패로 본다
```

## 주요 결과 요약

| 항목 | 실측 | 산출물 |
|---|---|---|
| 손역전파 vs 수치미분 | ŷ=**0.647292** · L=**0.434958** · ∂L/∂z₂=**−0.352708** · ∂L/∂W₁=[−0.043979, 0, −0.051734, 0] · 최대상대오차(0 제외) **2.041e-10** | `reports/backprop.json` |
| y_true 미정의 대응 | y=1.0 과 0.0 **양쪽** 손계산(같은 ŷ 에서 L 0.434958 vs 1.042114) | 〃 · `docs/y_true-definition.md` |
| det = 면적배 | 최대오차 **1.33e-11** · det=1 인 3행은 검사가 **항등식** → 축비 4.0 으로 대체 | `reports/linear_algebra.json` |
| power iteration | λ₂/λ₁ 0.9/0.5/0.1 → 반복 **105 / 19 / 7** (이론 ln(1e-12)/ln r ≈ 262/39/12 와 같은 자릿수) | `reports/power_iteration.json` |
| SVD 압축 | k=1/10/50 설명분산 93.95/99.53/99.99% 이나 저장비율 **3.1/31.5/157.5%** · k≥64(랭크 상한)는 개선 0 | `reports/svd.json` |
| 수치미분 ≤1e-4 | **x² 은 모든 h에서 통과**(최적오차 4.441e-16) → 요구 조건이 검증이 아님. x³·sin 은 실패로 갈린다. 최적 h 는 1e-1/1e-6/1e-5 로 제각각 | `reports/calculus.json` |
| 등고선 ⊥ 그라디언트 | 접선을 그라디언트로 유도하면 편차 **0.000°(항등식)**, 그려진 등고선에서 재면 **최대 0.8322°**(중위 0.0849°, 468점) | 〃 · `figures/03_gradient_contours.png` |
| lr 상한 | vanilla 측정 상한 **원 0.9 · 타원 0.09** vs 이론 2/λmax **1.0 · 0.1** | `reports/optimizer.json` |
| 모멘텀 | 타원 lr=0.15: vanilla **6.428e+61(발산)** / momentum **4.973e-03**(100회) → 1000회 **2.172e-44(수렴)** | 〃 |
| MSE=MLE · CE=MLE | 격자 0.001 위에서 argmin 차이 **0.000e+00** · 깁스는 참값 기준 성립(CE 0.647455 ≥ H 0.647447), 표본 플러그인은 **−0.0012 아래로** 왜곡 | `reports/probability.json` |
| softmax 안정성 | logits ×1000: 차감 없음 **nan** / 차감 있음 유한 · 원래 스케일에선 차이 1.11e-16(=검사 무의미) | 〃 |

**두 번 걸린 자기 검산 오류(노트북·리포트에 그대로 적었다)**: ① `numeric_grad` 의 loss 가 `forward` 인자를 한 칸 밀려 받아
모양은 맞되 값이 꼬였고, ② b1 섭동에 `EPS` 를 곱하지 않아 절대오차가 **4.8e3** 떴다. 둘 다 "역전파가 틀렸다"가 아니라
**검사 자체가 틀려서** 붉었다 → 그래서 검사에 '0이 아닌 항이 존재해야 한다'는 대조군을 넣었다.

## 계획(probe)에서 적었던 기대값과 다른 5건

| 항목 | 기대(probe) | 실측(구현) | 원인 |
|---|---|---|---|
| power iteration 반복수 | 86 / 22 / 7 | **105 / 19 / 7** | 판정 함수가 다르다(고윳값 수렴 vs Rayleigh quotient 상대변화 1e-12) |
| 타원 lr=0.15 vanilla | 1.100e+12 | **6.428e+61** | 반복 60회 → 100회(발산은 지수적으로 더 커진다) |
| momentum 동일 조건 | 4.601e-02 | **4.973e-03**(100회) | 시작점·판정 기준(반복수) 차이 → '판정은 기준과 함께' 규칙의 근거 |
| gradcheck 최대오차 | 8.6e-11 | **2.041e-10** | 1개 항 → 4개 항(dW1·dW2·db1·db2)의 최대로 확장 |
| SVD k=50 저장비율 | 156% | **157.5%** | 특이값 벡터 1개를 세는지의 차이 |

→ 규칙: 리포트에 적는 숫자는 **이 레포가 재생한 값**이고, 기대값은 '기대' 라벨로만 남긴다. 판정 기준(격자·tol·반복수)을
적지 않은 기대값은 비교 대상이 아니라 **다른 실험**이다.

## 요구 5조항 ↔ 검증 ↔ 판정

| # | 조항 | 이 레포의 증거 | 판정 근거 |
|---|---|---|---|
| 1 | 선형대수(변환·det·power·SVD) | `src/linear_algebra.py`·`power_iteration.py`·`svd_compress.py`, `figures/01,02` | 표에 '검사가 무의미한가' 열 포함 |
| 2 | 미적분(수치미분·등고선 직교) | `src/calculus.py`, `figures/03` | 항등식(전 h 통과)과 구분력(x³·sin 실패)을 함께 표기 |
| 3 | 역전파 손계산 | `notebooks/backprop_derivation.ipynb`(shape 표 포함), `src/backprop.py` | 4자리 일치 + 상대오차 ≤1e-7 + '0이 아닌 항' 검사 |
| 4 | 최적화(GD/Momentum) | `src/optimizer.py`, `figures/04,05` | 발산·수렴 동시 관측, 측정 상한 ≤ 이론 상한, 반복수 민감도 표 |
| 5 | 확률(MSE·CE = MLE) | `docs/probability_loss.md` + `notebooks/probability_loss.ipynb`, `src/probability.py`, `figures/06,07` | 격자 해상도 의존을 명시, 깁스는 모수 기준만 성립 |

## 갭 표 (명세 요구 vs 현재)

| 요구/결함 | 현재 | 부족분·대응 |
|---|---|---|
| 예제에 `y_true` 가 없다(명세 결함) | 양쪽 손계산 병기 + 표준을 y=1.0 으로 선언(`docs/y_true-definition.md`) | 채점자 의도와 다를 수 있다 → 두 값을 다 제출해 회피 |
| "중심차분 오차 ≤1e-4" | x² 에서는 어떤 h 에서도 통과 → **검사 무의미** | 대체 표(x³·sin·큰 h)를 넣고 리포트에 '무의미'를 명기 |
| "그림을 하나의 Figure에 겹쳐" 4건 | 01(3축)·04(경로 중첩)·05(함수별 상한)·02(원본+3복원) | 축 추가 대신 서브플롯으로 해결 |
| 보너스 Adam / Newton / 정보이론 | Adam 은 구현·스윕에 포함. **Newton·정보이론은 미수행** | 60h 대비 우선순위: 손계산 정합성 > 보너스. 미수행을 여기에 적어 둔다 |
| 실데이터·실이미지 | 없음(합성 64×64) | SVD·직교성 결론의 외적 타당도는 미검증 |
| G4(기대-실측 병기)는 **문서 형태 검사** | README·analysis 검증만 한다 | 값의 검증은 G1~G3 이 담당(강도 한계를 명시) |

## 게이트 (`reports/gates.json`)

G1 손계산 vs NumPy·수치미분 4자리(≤1e-7) · G2 검사 자체의 항등식 여부 표 · G3 lr 스윕 수렴/발산 + 모멘텀 ·
G4 기대-실측 병기 → **4/4 PASS**

## 규칙

1. 리포트에 적는 숫자는 이 레포가 실행해 만든 파일에서만 복사한다.
2. 계수·오차는 **재는 법(격자·h·반복수·함수)과 함께**만 쓴다. 조건 없는 "일치/발산"은 쓰지 않는다.
3. 검사가 통과만 한다면 대조군(반드시 실패해야 할 케이스)을 붙인다. 항등식으로 판명되면 리포트에 '무의미'로 적는다.
