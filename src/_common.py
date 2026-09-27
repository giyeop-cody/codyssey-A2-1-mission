"""A2-1 공용: 출력 디렉터리·폰트·그림 저장 검사(제목·축 라벨 누락 방지)."""
import json
import pathlib

import matplotlib

# 스크립트 실행(headless)에서는 Agg. **노트북 커널 안에서는 Agg 로 강제하지 않는다** —
# 강제하면 InlineBackend 가 꺼져서 plt.show() 가 image/png 출력을 내지 않는다(감사 반영:
# 이전 확률 노트북은 '분포 시각화' 요구를 텍스트 print 로만 대신했다).
try:  # pragma: no cover
    _ip = get_ipython()  # type: ignore[name-defined]  # noqa: F821
except NameError:
    _ip = None
if _ip is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_to_png_bytes(fig) -> bytes:
    """그림을 파일 없이 PNG 바이트로 뽑는다 — 노트북 출력에 그대로 embed 하는 데 쓴다.

    바이트를 반환하는 이유: IPython.display.Image(data=…) 는 str 을 **파일 경로**로 해석한다
    (base64 문자열을 넘기면 FileNotFoundError). 노트북에 그림을 남기려면 bytes 여야 한다.
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fig_to_png_b64(fig) -> str:
    import base64
    return base64.b64encode(fig_to_png_bytes(fig)).decode("ascii")
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
