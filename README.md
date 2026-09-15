# A2-1 — AI 학습을 수학으로 직접 풀어보기 (미션 제출물)

- 학습·계획·결정 기록: [codyssey-A2-1-studylog](https://github.com/giyeop-cody/codyssey-A2-1-studylog)
- 이 레포의 일정·게이트: [`docs/learning-map.md`](docs/learning-map.md) (hub에서 발췌·자동 생성)
- 과정 전체: [codyssey-A-studylog-hub](https://github.com/giyeop-cody/codyssey-A-studylog-hub)

## 실행

```bash
pip install -r requirements.txt
python -m pytest -q                 # 기본: 미구현 게이트는 skip
MISSION_STRICT=1 python -m pytest -q  # 제출 직전: skip도 실패로 본다
```

## 상태

- 태스크 11개 중 완료(☑) **0** · 진행(◐) **0** · 남은 것 **11**
- 게이트 3개 → `tests/test_gates.py` 가 1게이트 1함수다.

## 규칙

1. 리포트에 적는 숫자는 이 레포가 실행해 만든 파일에서만 복사한다.
2. 명세 최소치 미달은 감추지 않고 아래 표로 남긴다.
3. `git status`에 데이터(csv·원본 이미지)가 보이면 즉시 `.gitignore`로 보낸다.

## 갭 표 (명세 요구 vs 현재)

| 요구 | 현재 | 부족분 | 계획 |
|---|---|---|---|
| (구현하며 채운다. 비워두면 '미검증'으로 취급) | | | |

