# A2-1 평가 항목에 맞춘 산출물 개선

작성일: 2026-10-11

기존 제출물과 evalmap을 대조한 차이를 코드·산출물·재실행 근거로 보완했다. 성능이 낮은 대조군도 보고하고, 과제 조건의 수학적 충돌이나 실기기 미검증을 성공으로 바꾸지 않는다.

## 검증

`python -m pytest -q` — 23 passed, 14 warnings in 1.65s

실행 수치는 아래 재현 명령과 리포트에서 확인한다.

## Q1-10 — 조건 명확화·재검증

**문항 Q1-10**: lr=0.5 이상에서 발산 현상이 그래프로 확인되는가?

**수정 후 답변**: 함수별 Hessian과 Vanilla GD의 엄격한 안정 상한을 별도 러너로 검증했다. lr=.5, 12회에서 x²+y²는 반경 0이고 x²+10y²는 발산한다. 요구 문구의 수학적 조건을 명시하며 수렴 결과를 발산으로 바꾸지 않는다.

**근거**: [scripts/verify_lr_boundary.py](../scripts/verify_lr_boundary.py) · [reports/lr-boundary-verification.json](../reports/lr-boundary-verification.json) · [figures/09_lr_divergence.png](../figures/09_lr_divergence.png)
