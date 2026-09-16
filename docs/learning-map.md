# A2-1 미션 학습 맵 — AI 학습을 수학으로 직접 풀어보기

> 자동 생성(`hub/tools/gen_mission_repos.py`) · 원본: [studylog `docs/04-plan.md`](https://github.com/giyeop-cody/codyssey-A2-1-studylog/blob/main/docs/04-plan.md) ·
> 과정 전체 맵·태스크 보드는 [hub](https://github.com/giyeop-cody/codyssey-A-studylog-hub) `docs/02·03`. **이 파일은 발췌본이며 손으로 갱신하지 않는다.**

## 1. 단계 (시간·산출물·Done)

| 단계 | 시간 | 산출물 | Done |
|---|---|---|---|
| P0 수학 메모장 | 4h | docs/02 작성(용어 20개) | 모르는 개념을 문장으로 설명 가능 |
| P1 linear_algebra.py | 11h | 변환시각화·power_iteration·svd_compress | det/면적비 표, 반복 수 표, 저장비율 표 |
| P2 calculus.py | 7h | central_diff, gradient_field | h 스윕 표 + sin 병기 |
| P3 backprop notebook | 14h | 2-2-1 유도, 손계산+NumPy 일치 | 소수 4자리 일치, shape 전부 표기 |
| P4 optimizer.py | 10h | VanillaGD·Momentum·(Adam) | lr 스윕 표, 등고선 위 경로 1 Figure |
| P5 probability_loss | 7h | PDF/PMF/softmax, MSE·CE=MLE 유도 | 유도 4줄 이상, 실측 logLik |
| P6 README·정리 | 4h | 개요·실행법·결과요약 | 재현 테스트(클린 venv) |
| P7 자가채점·수정 | 3h | check 스크립트 ALL PASS | 요구 5항목 × 검수표 |
| 보너스 시간 과다 | P4 6h 초과 | Adam만 남기고 Newton·정보이론은 README "미수행 + 사유" |

## 2. 통과 게이트 — 이 숫자가 나오면 끝

| 미션 | 게이트 | 통과 조건(측정) | 근거 실측값 |
|---|---|---|---|
| A2-1 | G1 손계산 일치 | notebook 손계산 vs NumPy 4자리 | ŷ=**0.647292**, L=**0.434958**, ∂L/∂z₂=**−0.352708** |
| | G4 기대-실측 병기 | 계획(probe) 값과 구현 실측이 다른 행을 표로 두고 원인을 한 줄씩 적는다 | power 86/22/7→**105/19/7** · gradcheck 8.6e-11→**2.041e-10** · 격자·tol 차이 |
| | G2 검증 유효성 | 검사 자체의 항등식 여부를 표로 확인 | x²에서 중심차분 ≤5e-15 (모든 h) → 요구 1e-4는 **무의미** · 구현 실측 4.441e-16, x³·sin 은 실패로 갈린다 |
| | G3 옵티마이저 | lr 스윕 수렴/발산 + 모멘텀 비교 | x²+10y²: lr=0.15 vanilla **6.428e+61** 발산(계획 1.100e+12) / momentum **4.973e-03**(100회)→**2.172e-44**(1000회) · 상한 측정 0.09 vs 이론 0.1 |

## 3. 태스크 보드 (이 미션분)

| ID | 태스크 | 산출물 | 완료 기준 | h | 선행 | 상태 |
|---|---|---|---|---|---|---|
| A2-1-T1 | 용어·모르는 것 목록 | `docs/02` 20용어(30초 설명) | 각 용어를 "남에게 30초 설명" 문장화 | 3 | G1 | ☑ |
| A2-1-T2 | 선형대수 실습 | `linear_algebra.py`(변환 시각화) | det·면적비 표. **주의:** S(2,0.5)는 det=1 → 면적비 검사는 항등식. 다른 행렬로 대체 | 5 | T1 | ☑ |
| A2-1-T3 | 거듭제곱·power iteration | `power_iteration.py` | λ₂/λ₁ 비별 반복 수 표(기대 86/22/7 → 실측 **105/19/7**) | 3 | T2 | ☑ |
| A2-1-T4 | SVD 압축 | `svd_compress.py` | k=1/10/50 저장비율(3.1/31.2/**156%** — 순위 ≤64에선 압축이 역효과) | 4 | T2 | ☑ |
| A2-1-T5 | 수치미분 검증 설계 | `calculus.py::central_diff` | x²에선 모든 h에서 ≤5e-15 → **요구 1e-4 검사는 무의미**. sin·큰 h까지 포함한 대체 표 | 4 | T1 | ☑ |
| A2-1-T6 | 경사하강·모멘텀 | `optimizer.py` | lr 스윕 + x²+10y²에서 vanilla 발산(실측 **6.428e+61**) vs momentum 생존(100회 **4.973e-03**·1000회 2.172e-44), lr=0.50 양쪽 발산 | 7 | T5 | ☑ |
| A2-1-T7 | 손역전파 노트 | `backprop.ipynb` | 2-2-1 손계산 vs NumPy 4자리 일치(ŷ **0.647292** / L **0.434958** / ∂L/∂z₂ **−0.352708**) | 10 | T5 | ☑ |
| A2-1-T8 | `y_true` 정의 확정 | `docs/01` 결정 문단 | 정의서가 비어 있으므로 0/1 양쪽 손계산을 **둘 다** 싣고 선택 근거 명시 | 1 | T7 | ☑ |
| A2-1-T9 | MSE·CE = MLE 유도 | `probability_loss.md` | 유도 4줄↑ + 실측 로그우도 | 5 | T7 | ☑ |
| A2-1-T10 | gradcheck로 자체 검증 | `test_gradcheck.py` | rel.err ≤1e-7(기대 8.6e-11 → 실측 **2.041e-10**) | 3 | T7 | ☑ |
| A2-1-T11 | README·재현·자가채점 | README | 클린 venv 미실행(대신 **GitHub 클론 검증**으로 대체) + 검수표 47/47 | 4 | T10 | ◐ |

## 4. 위험 · Plan B (studylog 04-plan에서)

| 위험 | 대응 |
|---|---|
| (해당 없음) | |

## 5. 진행 규칙 (세 개만)

1. 검증 파일(테스트·표)을 **먼저** 만든다. `tests/` 가 이 레포의 1번 산출물이다.
2. 리포트에 적는 모든 숫자는 이 레포가 실행해서 만든 파일(`reports/*`)에서 복사한다. 예상치는 '예상'으로 적는다.
3. 명세 최소치에 못 미치면 숨기지 않고 갭 표를 README에 둔다.

