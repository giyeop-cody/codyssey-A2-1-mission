"""A2-1 공용: 출력 디렉터리·폰트·그림 저장 검사(제목·축 라벨 누락 방지)."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
REP, FIG = ROOT / "reports", ROOT / "figures"
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    assert fig.axes, f"{name}: 축이 없다"
    assert fig.axes[0].get_title(), f"{name}: 제목 누락"
    real = [a for a in fig.axes if getattr(a, "_colorbar", None) is None]
    for ax in real:
        assert ax.get_xlabel() or ax.get_ylabel(), f"{name}: 축 레이블 누락"
    assert any(ax.get_xlabel() for ax in real) and any(ax.get_ylabel() for ax in real), f"{name}: x·y 라벨"
    fig.savefig(FIG / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write(name: str, obj: dict) -> dict:
    REP.mkdir(parents=True, exist_ok=True)
    (REP / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return obj


def circle(n: int = 2001) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi, n)
    return np.stack([np.cos(t), np.sin(t)], axis=1)


def shoelace(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0
