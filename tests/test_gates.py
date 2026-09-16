"""A2-1 통과 게이트 테스트 — 자동 생성(hub/tools/gen_mission_repos.py).

각 함수의 docstring 이 곧 검수표다. 조건에 적힌 숫자는 학습기록 레포의 실측값이며,
판정은 **이 레포의 산출물**(reports/*.json, figures/*, *.md)에 대해 내린다.
"""
import json
import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import needs  # noqa: E402

def metric(name):
    f = ROOT / 'reports' / name
    return json.loads(f.read_text(encoding='utf-8')) if f.exists() else None

@needs()
def test_01_G1_notebook_손계산_vs_NumPy_4자():
    """[G1] notebook 손계산 vs NumPy 4자리 | 근거(학습기록 실측): """
    from conftest import gate_record
    gate_record("G1")

@needs()
def test_02_G4_계획_probe_값과_구현_실측이_다른_행을():
    """[G4] 계획(probe) 값과 구현 실측이 다른 행을 표로 두고 원인을 한 줄씩 적는다 | 근거(학습기록 실측): """
    from conftest import gate_record
    gate_record("G4")

@needs()
def test_03_G2_검사_자체의_항등식_여부를_표로_확인():
    """[G2] 검사 자체의 항등식 여부를 표로 확인 | 근거(학습기록 실측): """
    from conftest import gate_record
    gate_record("G2")

@needs()
def test_04_G3_lr_스윕_수렴_발산_모멘텀_비교():
    """[G3] lr 스윕 수렴/발산 + 모멘텀 비교 | 근거(학습기록 실측): """
    from conftest import gate_record
    gate_record("G3")

