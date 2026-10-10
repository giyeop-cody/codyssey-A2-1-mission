<!-- hand-authored -->
# A2-1 — AI 학습을 수학으로 직접 풀어보기 (제출물)


<!-- codyssey-links:start -->
## 🔗 Codyssey 연결

| 항목 | 링크 |
|---|---|
| **과제** | **A2-1** — AI가 어떻게 학습하는지 수학으로 직접 풀어보기 · 심화(Advanced) 「AI/SW 심화」 · AI 수학 기초 · 60h |
| 미션 원문 (정의서) | [`A2-1/a2-1-description.md`](https://github.com/giyeop-cody/codyssey-taskmap/blob/main/A2-1/a2-1-description.md) · [`A2-1-mission.jpg`](https://github.com/giyeop-cody/codyssey-taskmap/blob/main/A2-1/a2-1-mission.jpg) · [`meta.json`](https://github.com/giyeop-cody/codyssey-taskmap/blob/main/A2-1/meta.json) |
| 이 과제 연결 카드 | [`A2-1/links.md`](https://github.com/giyeop-cody/codyssey-taskmap/blob/main/A2-1/links.md) |
| 전체 연결 대장 | [`LINKS.md`](https://github.com/giyeop-cody/codyssey-taskmap/blob/main/LINKS.md) · 진행 현황 [`PROGRESS.md`](https://github.com/giyeop-cody/codyssey-taskmap/blob/main/PROGRESS.md) · [원문 API URL 41개](https://github.com/giyeop-cody/codyssey-taskmap/blob/main/codyssey-all-urls.md) |
| 과정 허브 | [codyssey-A-studylog-hub](https://github.com/giyeop-cody/codyssey-A-studylog-hub) `missions/A2-1/` · `repos/A2-1/` 서브모듈 |
| 통합 레포 | [codyssey](https://github.com/giyeop-cody/codyssey) → `A2-1/`(제출물) · `A2-1-studylog/`(학습기록) 서브모듈 |
| A→M 연결 | [codyssey-AI-Applied](https://github.com/giyeop-cody/codyssey-AI-Applied) '다음 단계 (M 과정 연결)' |
| 같은 과제의 다른 레포 | <sub>학습기록(스터디)</sub> [giyeop-cody/codyssey-A2-1-studylog](https://github.com/giyeop-cody/codyssey-A2-1-studylog) 🔒 |

> 🔒 = 비공개 레포. 상태·pin 커밋은 연결 카드와 `PROGRESS.md` 에 있다. 이 표는 2026-10-10 기준이며 미션 원문 3종은 원본 데이터라 진행 상태를 쓰지 않는다.
<!-- codyssey-links:end -->

## 프로젝트 개요

AI 학습의 다섯 가지 수학 기둥 — **선형대수·미적분·역전파·최적화·확률** — 을 라이브러리(autograd 등) 없이 NumPy 로 직접 구현하고,
각 결과를 원문이 정한 수치 기준(면적비 ≤1%, eig 비교 ≤5%, 수치미분 ≤1e-4, 반경 ≤0.1, 손계산↔NumPy 4자리, softmax 합 ±1e-6)으로 판정한다.

| 원문 제출물 | 이 레포 |
|---|---|
| 1. `src/linear_algebra.py` — 변환 시각화 · Power Iteration(λ와 고유벡터) · SVD 압축/복원 | `plot_transform` · `power_iteration` · `svd_compress`/`svd_reconstruct` (세 기능 모두 **이 한 파일**) |
| 2. `src/calculus.py` — 중심차분 수치미분 · Gradient 시각화 | `central_diff` · `numerical_gradient` · `gradient_visualization` |
| 3. `notebooks/backprop_derivation.ipynb` — 유도 · 손계산(체크포인트+shape) · NumPy 검증 | §1 순전파 식 → §2 출력층→입력층 연쇄 법칙 유도 → §3·§4 손계산 표(소수 6자리) → §5 노트북 내 독립 NumPy 와 4자리 대조 → §6 gradcheck → §7 y_true=0 |
| 4. `src/optimizer.py` — VanillaGD·Momentum 클래스 · 경로 시각화 | `VanillaGD` · `Momentum`(heavy-ball) · (보너스) `Adam` · 그림 04/05/09/10 |
| 5. `notebooks/probability_loss.ipynb` — PDF 2종·PMF 2종 · MSE/CE–MLE 유도 | §1 분포 그림(노트북에 embed) · §2 MSE=정규 MLE 유도 · §3 BCE=베르누이 · **카테고리 CE=카테고리 MLE** 유도 + 수치 확인 · §4 softmax |
| 6. README — 개요·실행 방법·주요 결과 요약 | 이 문서 |

- 학습·계획·결정·검증 기록: [codyssey-A2-1-studylog](https://github.com/giyeop-cody/codyssey-A2-1-studylog) (최신 재검증: `docs/08-verification-2026-10-11.md`, 평가 문항 답변: `docs/09-evaluation-answers.md`)
- 이 미션의 일정·게이트: [`docs/learning-map.md`](docs/learning-map.md) · 확률 유도 해설: [`docs/probability_loss.md`](docs/probability_loss.md) · y_true 미정의 대응: [`docs/y_true-definition.md`](docs/y_true-definition.md)
- 이 문서와 `reports/analysis.md` 의 숫자는 `reports/*.json`(=`scripts/run_all.py` 실행 결과)에서 옮겼다.

## 실행

```bash
pip install -r requirements.txt          # numpy · matplotlib · pytest · nbformat · nbclient · ipykernel
python scripts/run_all.py                # src 를 돌려 reports/*.json · analysis.md · 그림 14장 생성
python scripts/build_notebook.py         # 노트북 2종을 **실행해서** outputs(그림 포함)와 함께 저장 (run_all 다음에 실행)
python scripts/record_gates.py           # 게이트 G1~G4 판정 → reports/gates.json
python scripts/check_requirements.py     # 자가채점 [원문] 52 + [자체] 50 = 102건 (미달 시 exit 1)
python -m pytest -q                      # 23개 — 커밋된 산출물을 그대로 검증
MISSION_STRICT=1 python -m pytest -q     # 제출 직전: skip 도 실패로 본다
MISSION_BOOTSTRAP=1 python -m pytest -q  # 코드 수정 후: 산출물을 재생성해서 검증
```

재현 확인(2026-10-11): 새 venv(`python3 -m venv` → `pip install -r requirements.txt`)에서 위 순서 전체 실행 → 자가채점 102/102,
pytest 23 passed(기본·STRICT). 리포트 JSON·analysis.md·역전파 노트북은 커밋본과 바이트 동일(완전 복원 잔차 1e-26~1e-25 의 BLAS 잡음은 0 으로 절삭해 기록). 확률 노트북(텍스트 출력 동일)과 `figures/*.png` 는 PNG 바이트만 matplotlib 버전(3.10 vs 3.11)에 따라 다르다.

## 주요 결과 요약

| 원문 조항 | 실측 | 판정 · 해석 | 산출물 |
|---|---|---|---|
| 1-1~1-3 회전 R(30°)·S(2,0.5)·Sh(1.2) 를 단위 원(100점)에 적용 | 변환 전후 원 + meshgrid 격자점 + 기저벡터 e₁,e₂ 를 각 Figure에 겹침 | ✓ (원은 회전해도 원이므로 회전은 기저벡터 화살표로 보인다) | `figures/01a·01b·01c`, 요약 `01` |
| 1-4 det vs 면적비 ≤1% | 다각형 면적비 오차 최대 **4.4e-14%** · det 를 쓰지 않는 격자 세기(801×801) 추정 오차 최대 **0.011%** | ✓. 단 다각형 비율은 모든 M 에서 정리상 \|det\| 와 같아 부동소수 오차만 남는다 → 의미 있는 독립 측정은 격자 세기. det=1 인 세 행렬은 '변환을 빠뜨린 버그'와 구분이 안 돼 det=1.5 대조 행 · S(2,0.5) 축비 4.0 을 함께 둔다 | `reports/linear_algebra.json` |
| 1-5/1-6 Power Iteration vs `np.linalg.eig` ≤5% | λ₂/λ₁=0.9/0.5/0.1 → 반복 **105/19/7**(이론 132/20/6), λ 오차 최대 **4.0e-10%**, 고유벡터 각도차 최대 **3.6e-4°** | ✓. v0 가 최대 고유벡터에 직교하면 λ₂=**90** 을 반환(**실패 사례로 기록**) · 비대칭 대조(r=0.9)는 224회 | `figures/08` |
| 1-7 SVD k=10·50·100 (64×64 Grayscale 공개 이미지) | Grace Hopper 64×64(퍼블릭 도메인): k=10 PSNR **23.22 dB**·저장 **31.5%** / k=50 **52.14 dB**·**157.5%** / k=100 → 랭크 상한 64 = 완전 복원·**201.6%** | ✓(그림). 단 k=50·100 은 원본보다 **더 크게** 저장한다 — 압축은 k<31.75 에서만. Frobenius 오차 = Eckart–Young 예측 | `figures/02`, `02b` |
| 2-1 x², x=3, 해석값 6, ≤1e-4 | 오차 최대 **3.6e-8** | ✓. 그러나 f'''=0 이라 **모든 h** 에서 통과 — 이 함수로는 검사력이 없다(x³·sin 은 큰 h 에서 실패해 구분된다) | `reports/calculus.json` |
| 2-2~2-4 x²+y² 등고선 + ∇ 화살표 ⊥ | 화살표 = `numerical_gradient`, 등고선 위 점에 배치 · 그려진 등고선 실측 직교 편차 최대 **0.451°**(중위 0.029°, 1608점) · 타원 x²+4y² 대조 1.585° | ✓ | `figures/03` |
| 4-1~4-3 Vanilla GD, lr=0.1, 100회, 반경 ≤0.1, 경로 점선 | 최종 반경 **7.3e-10** · 그림 04 에 lr=0.01/**0.1**/0.15 경로, vanilla 점선 | ✓ | `figures/04` |
| 4-4 "lr≥0.5 에서 발산" | 원: 갱신식 x←(1−2lr)x → lr=0.5 는 **1회에 원점**, 0.5<lr<1 수렴(부호 반전), 1.0 진동, **>1 발산** · 타원: lr>0.1 발산 | 그래프 ✓(그림 09). **원문 문장은 x²+y² 에서는 사실이 아니다** — 타원 또는 lr>1 에서만 참. 대조 표로 기록 | `figures/09`, `05` |
| 4-5~4-7 Momentum(β=0.9) 비교 · 한 Figure 겹침 · 타원 이점 | 타원 lr=0.15: vanilla **6.4e+61(발산)** vs momentum **5.0e-3**(100회) · f<1e-6 도달: lr=0.01 vanilla 397 vs momentum 153 · 각자 최적: vanilla 45회 vs heavy-ball(β=0.27) 19회 | ✓. 이점은 **넓은 안정 lr 범위**(타원 vanilla <0.1 · momentum <0.19)와 작은 lr 에서의 가속. β=0.9 고정이면 각자 최적 lr 끼리는 vanilla 가 더 빠르다(수렴률 하한 √β≈0.949) — "모멘텀이 항상 빠르다"가 아니다 | `figures/04`, `10` |
| 3-1~3-7 역전파 (y_true 공란 → 1·0 양쪽) | ŷ=**0.647292** · y=1: L=**0.434958**, ∂L/∂ŷ=−1.544899, ∂L/∂z₂=**−0.352708**, ∂L/∂W₂=[−0.185165, −0.202611], ∂L/∂a₁=[−0.176354, −0.211625], ∂L/∂z₁=[−0.043979, −0.051734], ∂L/∂W₁=[[−0.043979, 0],[−0.051734, 0]] | ✓ 손계산 13항 ↔ 노트북 NumPy 4자리 일치(y=1·y=0 모두) · gradcheck 상대오차 ≤2.7e-10 · W₁ 둘째 열 0 은 x₂=0 때문 | 노트북, `reports/backprop.json` |
| 5-1 N(0,1)·N(2,0.5) | 정점 0.398942 / **0.797885**(σ=0.5 해석) | ✓. `N(μ,σ²)` 관례면 σ²=0.5(정점 0.564190) — 원문이 어느 쪽인지 밝히지 않아 노트북에 두 해석 대조 그림 | `figures/06`, 노트북 §1 |
| 5-2 B(0.3)·B(0.7) PMF | [0.7, 0.3] · [0.3, 0.7], 합 1 | ✓ | `figures/07` |
| 5-3 softmax 합=1 ±1e-6 | 최대 \|합−1\| **1.1e-16** (logits ×1000 은 max 차감 없이 nan) | ✓ | `reports/probability.json` |
| 5-4/5-5 MSE·CE = MLE 유도 (노트북) | 유도 LaTeX + 수치 확인: MSE argmin = 정규 MLE(기울기 1.486), BCE argmin = 베르누이 MLE(p̂ 0.348), 카테고리 CE argmin = 표본 빈도 [0.223, 0.485, 0.292] | ✓ · 깁스 CE≥H 는 참값 기준(0.647455 ≥ 0.647447)에서만, 표본 플러그인 H 는 −0.0012 편향 | 노트북 §2·§3 |
| 보너스 | Adam 구현·스윕 포함. Newton·정보이론(별도 구현)은 **미수행** | — | `src/optimizer.py` |

## 원문 자체의 모호·결함 (오해석 위험)

| 지점 | 위험 | 이 레포의 처리 |
|---|---|---|
| 역전파 예제 `정답: y_true =` 이 **공란** | 채점자의 y 와 다르면 L·모든 기울기가 다르다 | y=1(표준)·y=0 **둘 다** 손계산·NumPy 대조. y=1 은 관례적 선택(원문에 근거 숫자 없음) |
| "lr≥0.5 에서 발산" | x²+y² 에서 lr=0.5 는 한 번에 최솟값 — 문장을 믿고 '발산 그림'을 억지로 만들면 틀린다 | 실측 그대로 그리고(그림 09) 원/타원 대조 표 |
| N(2, 0.5) | σ 인지 σ² 인지 | σ=0.5 로 그리고 σ²=0.5 와 대조 |
| k=100 (64×64 이하 이미지) | 랭크 상한 64 를 넘는다 — k=100 은 k=64 와 동일 | 제목·표에 명시 |
| det vs 면적비 ≤1% | 다각형 넓이로 재면 정리상 항상 일치(검사력 없음) | 격자 세기 독립 추정 병기 |
| x² 중심차분 ≤1e-4 | f'''=0 이라 어떤 h 도 통과(검사력 없음) | x³·sin 대조 |
| Momentum 의 "이점" | lr·β 에 따라 결론이 뒤집힌다 | 같은 lr / 각자 최적 / 안정 범위를 분리해 정량화 |
| 평가 루브릭의 "반경 1 이내" | 원문(4-2)은 반경 **0.1** — 더 엄격한 원문 기준으로 판정(7.3e-10 이라 둘 다 통과) | studylog `docs/09` 에 기록 |

## 2026-10-11 재검증 정정 기록

"이전 상태는 원문에 **완전히** 부합하지 않았다"가 재검증 결론이다. 자가채점 82/82·pytest 통과였지만, 그 검사들은 파일·키워드
**존재**를 봤고 내용을 보지 않았다. 상세(발견 경위·근거·트러블슈팅)는 studylog `docs/08-verification-2026-10-11.md`.

| # | 발견 | 정정 |
|---|---|---|
| A | Power Iteration·SVD 가 `src/power_iteration.py`·`src/svd_compress.py` 에 있었다(원문: `src/linear_algebra.py` 한 모듈) · 함수가 고유벡터를 반환하지 않았다 | `linear_algebra.py` 로 통합, `(λ, v, 반복수, 이력)` 반환, 고유벡터를 eig 와 대조 |
| B | SVD 이미지가 합성 패턴(원문: 직접 촬영 또는 공개 이미지) | 공개 이미지 `assets/grace_hopper_64.png`(64×64 Grayscale) |
| C | 회전·스케일·전단이 한 그림의 서브플롯뿐, 회전은 원→원이라 **변화가 안 보였다** | 01a/01b/01c 개별 Figure + 격자점·기저벡터 |
| D | 면적비 ≤1% 검사가 정리상 항상 참(검사력 없음)인데 표는 'det=1 행만 무의미'로 표기 | 격자 세기 독립 추정 추가, 열 이름을 '무변환 버그 구분력 없음(det=1)'으로 |
| E | 등고선 직교 측정이 `allsegs[0]`(첫 레벨)만 보았다 · 화살표가 등고선 위가 아닌 격자 점에 있었다 | 전 레벨 측정(1608점), 화살표를 등고선 위 점에 `numerical_gradient` 로 |
| F | 그림 04 에 원문 지정 lr=0.1 경로가 없었고 vanilla 가 점선이 아니었다 (수정 1차본에서는 원 위의 두 경로가 같은 직선이라 momentum 이 vanilla 점선을 가렸다 — 그림을 보고 발견) | lr 0.01/0.1/0.15 · vanilla 점선을 맨 위 레이어로 |
| G | 그림 05 가 세로로 늘어나 곡선이 안 보이는 상태로 커밋돼 있었다(텍스트 좌표가 데이터 범위 밖) | 레이아웃 수정 + 자가채점에 그림 종횡비 검사 |
| H | 4-4 "lr≥0.5 발산"을 그래프로 보인 그림이 없었다 | 그림 09 |
| I | "모멘텀이 빠르다"가 lr·β 조건 없이 서술 | 그림 10 · 반복수 표(같은 lr / 각자 최적 / 안정 범위) |
| J | 역전파 노트북에 **단계별 유도가 없었고**, 역전파 6항은 손계산이 아닌 코드 값이었으며, "손계산↔NumPy 4자리 일치"는 실제로 **해석기울기↔수치미분(gradcheck)** 이었다 | LaTeX 유도 · 6자리 손계산 표 · 노트북 내 독립 NumPy 와 4자리 대조 · gradcheck 분리 |
| K | MSE/CE=MLE 유도가 노트북이 아닌 `docs/` 에만 있었고, **카테고리 분포 유도는 어디에도 없었다** | 노트북 §2·§3 에 유도 + 카테고리 수치 확인 |
| L | N(2,0.5) 의 σ/σ² 모호성을 밝히지 않았다 | 노트북 대조 그림 + 문서 |
| M | 깁스 문장 부등호 방향 오기("손실이 엔트로피보다 **크면** 이상하다" → **작으면**) | `src/probability.py`·`docs/probability_loss.md` |
| N | 베르누이 정정 기록의 이전 막대합이 README·docs 에 "0.6474/0.2492"로 적혀 리포트 기록값 **0.049429** 와 어긋났다 | 수치 정정(닫힌 식 `((1−p)^11−p^11)/(1−2p)` 로 확인) |
| O | `docs/y_true-definition.md` 가 "미션이 예시로 제시한 숫자(L=0.434958)"를 근거로 들었으나 원문엔 그런 숫자가 없다(순환 논증) | 관례적 선택임을 명시 |
| P | README 가 `requirements.txt` 에 nbformat·nbclient 가 있다고 적었지만 없었다(노트북 빌드 재현 불가) · pytest 개수 19 로 표기(실제 23) · 클린 venv 미검증 | requirements 보강, 클린 venv 재현 완료 |
| Q | 자가채점 82건이 위 결함을 하나도 잡지 못했다(존재·키워드 검사) | [원문] 검사를 내용 기반으로 교체·추가(52건), 이전 노트북·그림으로 되돌리면 실제로 FAIL 나는지 **변이 시험**으로 확인 |
| R | 원문 조항 번호 표기 오류: 역전파 shape 은 3-6, 4자리 일치는 3-7 인데 3-7/3-8 로 적혀 있었다 | 번호 정정 |

이전(2026-09-27 감사) 정정 기록 — 베르누이 PMF 를 진짜 베르누이로, N(0,3)→N(2,0.5), x=1→x=3, 타원→x²+y² 등고선, 모멘텀 안정 한계
2(1+β)/λmax, power iteration 이론식 ln tol/ln r² 등 — 은 커밋 `c9f435c` 의 README 와 studylog `docs/07` 에 남아 있다.

## 갭 표 (원문 요구 vs 현재)

| 요구 | 현재 | 남은 차이 |
|---|---|---|
| 1-1 "변환 전후를 하나의 Figure에 겹쳐" | 01a/01b/01c 각각 한 축에 겹침 | 없음 |
| 4-6 "두 옵티마이저 경로를 하나의 Figure에 겹쳐" | 그림 04 의 각 축에 vanilla(점선)·momentum(실선) 겹침 | 없음 |
| 4-4 "lr≥0.5 발산" | 원에서는 거짓 → 실측대로 기록 | 원문 문장과 결과가 다르다(원문 결함으로 판단) |
| 2-1 ≤1e-4 | 통과하지만 검사력 없음 | 대조 함수로 보완 |
| y_true 공란 | 양쪽 계산 | 채점 기준이 다른 y 를 가정해도 비교 가능 |
| 보너스 Newton · 정보이론 | 미수행 | 선택 과제 — 수행했다고 쓰지 않는다 |
| 외적 타당도 | 이미지 1장(64×64) | 다른 이미지에서 PSNR/저장비율 곡선은 달라진다 |

## 게이트 (`reports/gates.json`)

| 게이트 | 판정 대상(원문 조항) |
|---|---|
| G1 | 3-4 순전파 중간값 · 3-5 필수 6항 · 3-7 노트북 손계산↔NumPy 4자리(y=1·0) · gradcheck ≤1e-7 |
| G2 | 2-1 x=3·해석값 6·≤1e-4 · 2-2 등고선 x²+y² · 2-4 직교 편차 ≤1° (+ 항등식 표기·구분력) |
| G3 | 4-2 반경 ≤0.1 · 4-6/4-7 모멘텀 이점(타원 lr=0.15) · 측정 상한 ≤ 이론 상한 |
| G4 | 원문 조항 게이트 6종(1-4·1-6·1-7·3-5·5-2·5-3) + 리포트/README 기대-실측·정정 기록 |

## 구조

```
src/        _common(출력·폰트·그림검사) · linear_algebra(변환·power iteration·SVD) · calculus · optimizer · backprop · probability
scripts/    run_all(파이프라인+analysis.md) · build_notebook(실행해서 저장) · _nb_backprop · _nb_probability(노트북 셀 정의)
            record_gates · check_requirements
tests/      test_gates · test_math · test_gradcheck · test_self_check · test_repo_shape  (23개)
notebooks/  backprop_derivation · probability_loss(그림 4장 embed)
assets/     grace_hopper_64.png (SVD 입력, 퍼블릭 도메인)
reports/    *.json 9종 + analysis.md     figures/ 01·01a–c·02·02b·03–10 (14장)     docs/ learning-map · probability_loss · y_true-definition
```
