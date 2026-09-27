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

> 🔒 = 비공개 레포. 상태·pin 커밋은 연결 카드와 `PROGRESS.md` 에 있다. 이 표는 2026-09-27 기준이며 미션 원문 3종은 원본 데이터라 진행 상태를 쓰지 않는다.
<!-- codyssey-links:end -->

- 학습·계획·결정 기록: [codyssey-A2-1-studylog](https://github.com/giyeop-cody/codyssey-A2-1-studylog)
- 이 미션의 일정·게이트: [`docs/learning-map.md`](docs/learning-map.md) · 확률 유도: [`docs/probability_loss.md`](docs/probability_loss.md) · 예제 정의: [`docs/y_true-definition.md`](docs/y_true-definition.md)
- 과정 전체: [codyssey-A-studylog-hub](https://github.com/giyeop-cody/codyssey-A-studylog-hub)
- 이 레포에 적힌 모든 숫자는 `reports/*.json` 에 있고, 그 파일은 `scripts/run_all.py` 가 만든다(손으로 채운 숫자는 없다).
- **2026-09-27 비판적 감사**를 받아 원문 스펙 이탈 8건·서사 오류 9건·자가채점 결함 7건을 정정했다 → 아래 [감사 반영 기록](#2026-09-27-감사-반영-정정-기록).

## 실행

```bash
pip install -r requirements.txt          # numpy·matplotlib·pytest·nbformat·nbclient·ipython
python scripts/run_all.py                # src 7종을 돌려 reports/*.json·analysis.md·그림 8장 생성
python scripts/build_notebook.py         # 노트북 2종을 **실행해서** outputs(그림 포함)와 함께 저장
python scripts/record_gates.py           # 게이트 판정 → reports/gates.json
python scripts/check_requirements.py     # 원문 조항 36 + 자체 기준 46 자가채점 (미달 시 exit 1)
python -m pytest -q                      # 기본 (커밋된 산출물을 그대로 검증한다)
MISSION_STRICT=1 python -m pytest -q     # 제출 직전: skip 도 실패로 본다
MISSION_BOOTSTRAP=1 python -m pytest -q  # 코드 수정 후: 산출물을 재생성해서 검증한다
```

## 주요 결과 요약

| 항목 | 실측 | 산출물 |
|---|---|---|
| 손역전파 vs 수치미분 | ŷ=**0.647292** · L=**0.434958** · ∂L/∂y_pred=**−1.544899** · ∂L/∂z₂=**−0.352708** · ∂L/∂a₁=[−0.176354, −0.211625] · ∂L/∂W₁=[−0.043979, 0, −0.051734, 0] · 최대상대오차(0 제외) **2.041e-10** | `reports/backprop.json` |
| y_true 미정의 대응 | y=1.0 과 0.0 **양쪽** 손계산(같은 ŷ 에서 L 0.434958 vs 1.042114) | 〃 · `docs/y_true-definition.md` |
| det = 면적배 (원문 1-4, ≤1%) | 최대 상대오차 **1.333e-07%** · 통과. 단 det=1 인 3행은 검사가 **항등식** → 축비 **4.0** 을 **별도 열**로 대체 | `reports/linear_algebra.json` |
| power iteration (원문 1-6, `np.linalg.eig` 비교 ≤5%) | λ₂/λ₁ 0.9/0.5/0.1 → 반복 **105 / 19 / 7** · eig 비교 최대상대오차 **4.0e-10%** · 이론 예측(Rayleigh 기준 `ln tol/ln r²`) **132/20/6** | `reports/power_iteration.json` · `figures/08` |
| 직교 초기벡터 케이스 | **실패 사례로 기록** — λ_max(100) 대신 λ₂(**90.0**) 반환, 상대오차 10% | 〃 |
| SVD 압축 (원문 1-7, k=10·50·100) | 설명분산 99.53/99.99/100% 이나 저장비율 **31.5/157.5/201.6%** · k=100 은 랭크 상한(64) 때문에 k=64 와 동일 | `reports/svd.json` · `figures/02` |
| 수치미분 (원문 2-1: **x=3**, 해석값 6, ≤1e-4) | x² 오차 최대 **3.646e-08** → 통과. 그러나 f'''=0 이라 **모든 h 에서 통과** → 요구 조건이 검증이 아님. x³·sin 은 실패로 갈린다(최적 h 1e-5/1e-5) | `reports/calculus.json` |
| 등고선 ⊥ 그라디언트 (원문 2-2 함수 **x²+y²**) | 그려진 등고선 실측 최대 **0.1878°**(중위 0.0249°, 616점) · 유도 접선은 **0.000°(항등식)**. 타원 x²+4y² 대조는 최대 0.8322° | 〃 · `figures/03` |
| 원문 4-2 (lr=0.1, 100회, 반경 ≤0.1) | 최종 반경 **7.345e-10** → 통과 | `reports/optimizer.json` |
| lr 상한 | vanilla 측정 상한 **원 0.9 · 타원 0.09** vs 이론 `2/λmax(H)` **1.0 · 0.1** | 〃 |
| 모멘텀 이론 상한 (정정) | heavy-ball(v←βv+g) 은 실효 lr = lr/(1−β)=10·lr → `2(1−β)/λmax(H)` = **원 0.1 · 타원 0.01** (이전엔 vanilla 와 같은 값으로 잘못 표기) | 〃 |
| 모멘텀 이점 | 타원 lr=0.15: vanilla **6.428e+61(발산)** / momentum **4.973e-03**(100회, 원문 기준 반경 통과) → 1000회 **2.172e-44(수렴)** | 〃 · `figures/04` |
| 원문 4-4 "lr≥0.5 발산" | **원(x²+y²)에서는 성립하지 않는다** — vanilla lr=0.5 → f 0.000e+00(수렴), momentum → 2.369e-05. 타원에서는 발산(2.822e+192). 이론 상한이 원 1.0 / 타원 0.1 이므로 원문 문장은 타원에만 해당 | 〃 `원문4-4_대조` |
| MSE=MLE · CE=MLE | 격자 0.001 위에서 argmin 차이 **0.000e+00** · 깁스는 참값 기준 성립(CE 0.647455 ≥ H 0.647447), 표본 플러그인은 **−0.0012 아래로** 왜곡 | `reports/probability.json` |
| 원문 5-1 정규 PDF | **N(0,1)**(peak 0.398942) vs **N(2,0.5)**(peak 0.797885), 하나의 Figure | `figures/06` |
| 원문 5-2 베르누이 PMF | **B(0.3)=[0.7, 0.3] · B(0.7)=[0.3, 0.7]**, 합 **1.0**(오차 0) | `figures/07` |
| 원문 5-3 softmax 합=1 | 최대 \|합−1\| **1.110e-16** ≤ 1e-6 → 통과(세 스케일) | `reports/probability.json` |
| softmax 안정성 | logits ×1000: 차감 없음 **nan** / 차감 있음 유한 · 원래 스케일에선 차이 1.11e-16(=검사 무의미) | 〃 |

**두 번 걸린 자기 검산 오류(노트북·리포트에 그대로 적었다)**: ① `numeric_grad` 의 loss 가 `forward` 인자를 한 칸 밀려 받아
모양은 맞되 값이 꼬였고, ② b1 섭동에 `EPS` 를 곱하지 않아 절대오차가 **4.8e3** 떴다. 둘 다 "역전파가 틀렸다"가 아니라
**검사 자체가 틀려서** 붉었다 → 그래서 검사에 '0이 아닌 항이 존재해야 한다'는 대조군을 넣었다.

## 계획(probe)에서 적었던 기대값과 다른 5건

| 항목 | 기대(probe) | 실측(구현) | 원인 |
|---|---|---|---|
| power iteration 반복수 | 86 / 22 / 7 | **105 / 19 / 7** | 판정 함수가 다르다(고윳값 수렴 vs Rayleigh quotient 상대변화 1e-12). 이론 예측도 그 기준에 맞춰 `ln tol/ln r²` = **132/20/6** 으로 정정했다 |
| 타원 lr=0.15 vanilla | 1.100e+12 | **6.428e+61** | 반복 60회 → 100회(발산은 지수적으로 더 커진다) |
| momentum 동일 조건 | 4.601e-02 | **4.973e-03**(100회) | 시작점·판정 기준(반복수) 차이 → '판정은 기준과 함께' 규칙의 근거 |
| gradcheck 최대오차 | 8.6e-11 | **2.041e-10** | 1개 항 → 4개 항(dW1·dW2·db1·db2)의 최대로 확장 |
| SVD k=50 저장비율 | 156% | **157.5%** | 특이값 벡터 1개를 세는지의 차이 |

→ 규칙: 리포트에 적는 숫자는 **이 레포가 재생한 값**이고, 기대값은 '기대' 라벨로만 남긴다. 판정 기준(격자·tol·반복수)을
적지 않은 기대값은 비교 대상이 아니라 **다른 실험**이다.

## 요구 5조항 ↔ 검증 ↔ 판정

| # | 조항 | 이 레포의 증거 | 원문 조항 게이트 |
|---|---|---|---|
| 1 | 선형대수(변환·det·power·SVD) | `src/linear_algebra.py`·`power_iteration.py`·`svd_compress.py`, `figures/01,02,08` | 1-4 면적비 ≤1% ✓ · 1-6 eig 비교 ≤5% ✓ · 1-7 그림 k=10·50·100 ✓ |
| 2 | 미적분(수치미분·등고선 직교) | `src/calculus.py`(`central_diff`·`gradient_visualization`), `figures/03` | 2-1 x=3·해석값 6·≤1e-4 ✓ · 2-2 등고선 x²+y² ✓ · 2-4 직교 편차 ≤1° ✓ |
| 3 | 역전파 손계산 | `notebooks/backprop_derivation.ipynb`(shape 표 포함), `src/backprop.py` | 3-4 중간값 ✓ · 3-5 필수 6항(∂L/∂y_pred·z₂·W₂·a₁·z₁·W₁) ✓ · 3-7 shape ✓ · 3-8 4자리 일치 ✓ |
| 4 | 최적화(GD/Momentum) | `src/optimizer.py`, `figures/04,05` | 4-2 반경 ≤0.1 ✓ · 4-6 경로 겹침 ✓ · 4-7 모멘텀 이점 ✓ · 4-4 는 원문과 실측이 충돌 → 대조 표로 기록 |
| 5 | 확률(MSE·CE = MLE) | `docs/probability_loss.md` + `notebooks/probability_loss.ipynb`(그림 3장 embed), `src/probability.py`, `figures/06,07` | 5-1 N(0,1)·N(2,0.5) ✓ · 5-2 B(0.3)·B(0.7) 합=1 ✓ · 5-3 softmax 합=1 ±1e-6 ✓ · 5-4/5-5 유도 ✓ |

자가채점: `python scripts/check_requirements.py` → **원문 조항 36 + 자체 기준 46 = 82건**.
[원문] 태그가 미션 원문 조항을 직접 판정하는 항목이고, [자체] 는 이 레포가 세운 기준(검사의 유효성·문서 형태)이다.
[자체] 중 문서 형태 검사는 "문장에 그 단어가 있는가"만 본다(강도가 낮다) — 값 검증은 [원문] 항목과 `tests/` 가 담당한다.

## 2026-09-27 감사 반영 (정정 기록)

비판적 감사로 **원문 스펙 이탈·서사 오류·자가채점 결함**을 찾아 정정했다. 정정은 "조용히 고침"이 아니라
이 표와 리포트 안의 정정 기록(`probability.json` → `베르누이PMF_정정_기록`, `optimizer.json` → `이론_상한_정의`)으로 남긴다.

### 정정 1 — 원문 스펙 이탈 8건 (전부 미공개였음 → 지금은 원문대로 구현 + 대조를 명시)

| 원문 조항 | 이전 구현 | 지금 |
|---|---|---|
| 5-2 베르누이 **B(0.3)·B(0.7) PMF** | `p^k(1−p)^(10−k)`, k=0..10 을 "베르누이 PMF" 라벨로 그림 → 막대 합 **0.6474/0.2492 ≠ 1**, 이항계수 없음 = **확률분포가 아님** | k∈{0,1} 의 진짜 베르누이 PMF([0.7,0.3]·[0.3,0.7], 합 1.0). 배우려던 Binomial(10,p) 은 **이름을 바로 적어** 노트북에 대조 그림으로 분리 |
| 5-1 정규 **N(0,1) vs N(2,0.5)** | N(0,1) vs **N(0,3)** | 원문 두 분포를 단일 축 Figure에 겹침 |
| 2-1 수치미분 **x=3**, 해석값 6 | **x=1.0** 에서 측정(조건란에만 기재) | x=3 을 기본으로, x=1.0 은 민감도 대조로 강등 |
| 2-2 등고선 **x²+y²** | **x²+4y²**(미기재) | x²+y² 를 원문 행으로, 타원은 '추가(타원)' 대조 행으로 |
| 1-2 **S(2,0.5)** 그림 | 그림에 없고 표에만 | `figures/01` 을 4축으로 확장해 S(2,0.5) 를 변환 전후 겹침으로 출력 |
| 1-5/1-6 power iteration **시각화** | 그림 없음(표만) | `figures/08_power_iteration.png`(Rayleigh 오차 수렴곡선 + 정지점 + 이론 예측) |
| 1-7 SVD **k=10·50·100** 그림 | 그림은 k=1,10,50,**64** | 원문 k 3종 + 대조 k=1 로 변경, k=100 이 랭크 상한(64)과 같다는 사실을 제목에 기재 |
| 제출물 5 노트북의 **분포 시각화** | 노트북에 이미지 출력 0건(print만) | 노트북 안에서 그린 그림 3장을 `display_data(image/png)` 로 embed |

### 정정 2 — 수학·서사 오류

1. **잠복 버그(감사가 드러낸 것)**: `FUNCS["x^2"]` 의 도함수가 `lambda x: 2.0 + 0*x`(상수 2)였다. x=1 에서는 맞지만
   **x=3 으로 옮기면 해석값이 2로 남아 오차가 4.0** 이 된다 → `2.0*x` 로 정정. 스펙을 원복하지 않았다면 영원히 안 드러났을 오류다.
2. **모멘텀 이론 상한**: 세 옵티마이저 모두에 `2/λmax` 를 적었다. heavy-ball(v←βv+g)은 실효 lr이 lr/(1−β)=10·lr 라
   상한이 `2(1−β)/λmax(H)` = 타원 **0.01** 이다. 그리고 코드 안 `λmax` 는 `eigvalsh(A)[-1]`=2·λmax(H) 였다(숫자는 우연히 맞았으나 표기가 인수와 어긋남) → 표기를 헤세 기준으로 통일.
3. **power iteration 이론 공식**: `ln(tol)/ln(r)`(=262/39/12) 은 고유벡터 오차 기준. 실제 수렴 판정은 Rayleigh 변화 1e-12 이므로
   `ln(tol)/ln(r²)`(=**132/20/6**)이 맞다. "같은 자릿수" 주장도 이전 공식 기준으로는 2.5배 차이였다.
4. **직교 초기벡터 행**: λ₂=90 을 반환한 **실패 사례**인데 상대오차 1.579e-14% 만 강조돼 성공처럼 읽혔다 → `판정` 열에 실패임을 명시, 조건란도 행별 v0 로 분리.
5. **판정 라벨 왜곡**: `f<1e-8` 미만을 전부 "미수렴(**진동**)"이라 붙였다. 원 lr=0.01·0.99 는 부호 반전 없는 단조 감소였다 →
   발산 / **진동(좌표 부호반전 관측 시에만)** / 정체(천천히 감소) / 수렴 4분할 + **원문 기준(반경 ≤0.1)** 열 추가.
6. **`analysis.md` 의 과장**: "미션이 요구한 '하나의 Figure에 겹쳐'를 **모두 지켰다**" → 실제로는 01·02·03·05 가 서브플롯이다.
   지금도 01(4축)·02(4축)·03(2축)·05(2축)는 서브플롯이고, 같은 축에 겹친 것은 04·06·08 이다 → 사실을 그대로 적는다(갭 표 참조).
7. **면적비 열에 축비 값**: `linear_algebra.json` 의 '면적비(실측)' 열에 축비 4.0 이 들어 있었다(자체 원칙 "열 이름=측정 정의" 위반) → `축비(실측)` 별도 열로 분리.
8. **`docs/probability_loss.md` 의 미실측 주장**: "격자를 0.1 로 굵히면 두 argmin 이 어긋난다" → 같은 격자 위 두 목적함수는 단조관계라 갈리지 않는다. 실측도 없는 문장이므로 삭제하고, 격자 해상도가 규정한 것은 **일치 정밀도**임을 적었다.
9. **원문 4-4 와의 충돌을 침묵**: "lr≥0.5 발산"은 원에서 거짓이다(이론 상한 1.0). 이전엔 learning-map 의 "lr=0.50 양쪽 발산" 한 줄로 뭉갰다 → `optimizer.json` 의 `원문4-4_대조` 표와 analysis.md 절로 명시.

### 정정 3 — 자가채점 장치

| 이전 | 지금 |
|---|---|
| `check_requirements.py` 47건 **전부 자체 기준** — 원문 조항(N(2,0.5)·PMF 합=1·x=3·x²+y²·S(2,0.5) 그림·k=10/50/100·softmax 합=1·eig 비교)은 **0건** 검증 | **[원문] 태그 36건 신설**(원문 조항을 값으로 판정) + [자체] 46건. 출력을 "원문 X/X · 자체 Y/Y" 로 분리하고, 형태 검사의 강도 한계를 마지막 줄에 적는다 |
| 리포트 JSON 이 없으면 **조용히 건너뜀** → 누락이 통과로 흡수 | JSON 누락은 **FAIL** 로 센다 |
| `tests/test_gates.py` 4개 테스트에 **assert 없음**(게이트는 gates.json 자가 선언) | 각 테스트가 gates.json 선언 + **reports/*.json 값을 직접 재판정**(둘이 어긋나면 붉어짐) · 원문 분포·함수·그림을 확인하는 테스트 1건 추가 |
| `conftest.ensure_inputs()` 가 매 테스트 전에 run_all·record_gates 를 **항상** 실행 → 커밋된 산출물이 틀려도 스스로 재생성해 통과(자가치유) | 산출물이 **없을 때만** 생성. `MISSION_BOOTSTRAP=1` 을 줘야 재생성한다 → 기본 실행은 **커밋된 산출물**을 검증 |
| G2 의 통과 조건이 "x²는 전 h 통과(=요구가 무의미하다)" — **결론이 통과조건인 순환** | G2 는 원문 2-1(x=3·해석값 6·≤1e-4)·2-2(x²+y²)·2-4(직교 ≤1°)를 판정하고, '무의미함'은 게이트가 아니라 리포트 판독으로 남긴다 |
| G3 의 모멘텀 판정 `f<1.0`(13→0.5 도 통과) | 원문 기준(반경 ≤0.1) + 1000회 수렴 + **모멘텀 이론상한 0.01** 까지 판정 |

### 남은 것 (정정하지 않음 — 이유와 함께)

아래는 [갭 표](#갭-표-명세-요구-vs-현재)에 그대로 두고, "지켰다"고 쓰지 않는다.

## 갭 표 (명세 요구 vs 현재)

| 요구/결함 | 현재 | 부족분·대응 |
|---|---|---|
| 예제에 `y_true` 가 없다(**원문 결함**) | 양쪽 손계산 병기 + 표준을 y=1.0 으로 선언(`docs/y_true-definition.md`) | 채점자 의도와 다를 수 있다 → 두 값을 다 제출해 회피 |
| 원문 4-4 "lr≥0.5 발산"이 **원에서 거짓**(원문 결함) | `optimizer.json` → `원문4-4_대조` 표 + analysis.md 절 | 원문 문장을 따르지 않고 **측정 결과를 우선**한다(타원에서는 성립) |
| "중심차분 오차 ≤1e-4" | x=3 에서도 x² 은 모든 h 통과 → **검사 무의미** | 대체 표(x³·sin·큰 h)를 넣고 리포트에 '무의미'를 명기 |
| "그림을 하나의 Figure에 겹쳐" 4건 | **같은 축 겹침**: 04(경로)·06(PDF)·08(수렴곡선) / **서브플롯**: 01(4축)·02(4축)·03(2축)·05(2축) | 서브플롯은 '하나의 Figure'는 만족하나 '겹쳐'는 아니다 → 그대로 적는다 |
| 보너스 Adam / Newton / 정보이론 | Adam 은 구현·스윕에 포함. **Newton·정보이론은 미수행** | 60h 대비 우선순위: 원문 조항 정합성 > 보너스. 미수행을 여기에 적어 둔다 |
| 실데이터·실이미지 | 없음(합성 64×64) | SVD·직교성 결론의 외적 타당도는 미검증 |
| `Gradient 시각화 함수`(제출물 2) | `calculus.py::gradient_visualization()` — 독립 모듈이 아니라 등고선 그림과 한 파일 | 형태는 원문과 다르나 기능(∇f 계산 + 화살표 출력)은 충족 |
| G4(원문 조항 게이트 존재·통과 + 기대-실측 병기)는 **문서 형태 검사가 섞여 있다** | 값 게이트 6종 + 문서 키워드 검사 | 강도 한계를 gates.json `how` 와 자가채점 마지막 줄에 명시 |
| 클린 venv 재현 검증 | 미실행(이 레포는 GitHub 클론 후 `run_all` 재생성으로 대체 확인) | 학습맵 T11 을 ◐ 로 두고 **완료로 표시하지 않는다** |

## 게이트 (`reports/gates.json`)

| 게이트 | 판정 대상(원문 조항) |
|---|---|
| G1 | 3-4 순전파 중간값 · 3-5 필수 유도 6항 · 3-8 손계산↔NumPy 4자리 · gradcheck ≤1e-7 |
| G2 | 2-1 x=3·해석값 6·≤1e-4 · 2-2 등고선 x²+y² · 2-4 직교 편차 ≤1° (+ 항등식 표기·구분력) |
| G3 | 4-2 반경 ≤0.1 · 4-6/4-7 모멘텀 이점(타원 lr=0.15) · 측정 상한 ≤ 이론 상한(모멘텀은 heavy-ball 공식) |
| G4 | 원문 조항 게이트 6종(1-4·1-6·1-7·3-5·5-2·5-3)의 존재·통과 + 리포트/README 의 기대-실측·정정 기록 |

## 구조

```
src/        _common(출력·폰트·그림검사) · linear_algebra · power_iteration · svd_compress
            calculus · optimizer · backprop · probability
scripts/    run_all(파이프라인+analysis.md) · build_notebook(실행해서 저장) · record_gates · check_requirements
tests/      test_gates(게이트+값 재판정) · test_math(불변량) · test_gradcheck · test_self_check · test_repo_shape
notebooks/  backprop_derivation · probability_loss(그림 3장 embed)
reports/    *.json 8종 + analysis.md   figures/ 01~08   docs/ learning-map · probability_loss · y_true-definition
```
