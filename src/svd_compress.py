"""요구 1-3: SVD로 이미지를 압축하면 얼마나 잃고 얼마나saved 하나.

64×64 는 랭크 상한이 64이므로 **k=100 은 존재하지 않는다**(k=min(m,n)=64 이상은 항등 반복) —
미션이 예로 든 k=100 를 그대로 돌리면 저장비율이 100%를 넘는 '압축의 역효과'가 보인다.
"""
from __future__ import annotations

import numpy as np

from _common import save, write


def synth(n: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(float)
    img = 128 + 60 * np.sin(2 * np.pi * x / 11.0) * np.cos(2 * np.pi * y / 17.0)
    return np.clip(img + rng.normal(0, 12, (n, n)), 0, 255)


def svd_compress(img: np.ndarray, k: int):
    U, S, Vt = np.linalg.svd(img, full_matrices=False)
    kk = min(k, len(S))
    rec = (U[:, :kk] * S[:kk]) @ Vt[:kk]
    mse = float(np.mean((img - rec) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    m, n = img.shape
    stored = kk * (m + n + 1)
    return rec, {"k": k, "실제 랭크": kk, "mse": f"{mse:.3e}", "psnr_dB": round(psnr, 2),
                 "저장비율_%": round(stored / (m * n) * 100, 1),
                 "설명분산_%": round(float(np.sum(S[:kk] ** 2) / np.sum(S ** 2) * 100), 2)}


def main() -> dict:
    img = synth()
    rows = [svd_compress(img, k)[1] for k in (1, 5, 10, 30, 50, 64, 100)]
    fig, axs = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots(1, 4, figsize=(11, 3.0))
    for ax, (k, rec) in zip(axs, [(1, svd_compress(img, 1)[0]), (10, svd_compress(img, 10)[0]),
                                  (50, svd_compress(img, 50)[0]), (64, svd_compress(img, 64)[0])]):
        ax.imshow(rec, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"k={k}")
        ax.set_xlabel("열"); ax.set_ylabel("행")
    save(fig, "02_svd_compare.png")
    out = {"표": rows, "조건": "64×64 float64 합성 이미지(사인파+가우시안 σ=12), 시드 42, np.linalg.svd full_matrices=False",
           "판정": "k≥64 는 랭크 상한이라 더 저장해도 개선 0 → 저장비율이 100%를 넘는 구간이 '압축 실패'로 보인다"}
    return write("svd.json", out)


if __name__ == "__main__":
    d = main()
    for r in d["표"]:
        print(r)
    print(d["판정"])
