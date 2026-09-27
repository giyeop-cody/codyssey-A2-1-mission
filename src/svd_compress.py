"""요구 1-3(선형대수): SVD로 이미지를 압축하면 얼마나 잃고 얼마나 저장하나.

원문 1-7 은 **k=10, 50, 100** 의 복원 결과를 하나의 Figure에 비교하라고 한다.
64×64 는 랭크 상한이 64 이므로 k=100 은 k=64 와 **같은 그림**이 된다 — 그래도 원문 k 를 그대로 그려서
"k≥랭크에서는 개선이 0"임을 그림으로도 보이고, 제목에 랭크 상한을 적는다(감사 반영: 이전 그림은 k=1,10,50,64 였다).
k=1 은 압축이 실제로 이득인 구간을 보여주는 대조로 남긴다.
"""
from __future__ import annotations

import numpy as np

from _common import plt, save, write

KS_SPEC = (10, 50, 100)              # 원문 1-7
KS_TABLE = (1, 5, 10, 30, 50, 64, 100)
KS_FIG = (1, 10, 50, 100)            # 원문 k 3종 + 대조 k=1


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
                 "설명분산_%": round(float(np.sum(S[:kk] ** 2) / np.sum(S ** 2) * 100), 2),
                 "k_요청_대비_랭크_상한": ("랭크 상한(=min(m,n)) 초과 → k=64 와 동일" if k > kk else "유효")}


def main() -> dict:
    img = synth()
    rows = [svd_compress(img, k)[1] for k in KS_TABLE]
    spec_rows = [r for r in rows if r["k"] in KS_SPEC]

    fig, axs = plt.subplots(1, len(KS_FIG), figsize=(11.6, 3.1))
    for ax, k in zip(axs, KS_FIG):
        rec, meta = svd_compress(img, k)
        ax.imshow(rec, cmap="gray", vmin=0, vmax=255)
        note = " (=k=64, 랭크 상한)" if meta["실제 랭크"] < k else ""
        ax.set_title(f"k={k}{note}\nmse {meta['mse']} · 저장 {meta['저장비율_%']}%", fontsize=8)
        ax.set_xlabel("열"); ax.set_ylabel("행")
    axs[0].set_ylabel("행")
    for a in axs[1:]:
        a.set_ylabel("")
    save(fig, "02_svd_compare.png")

    out = {"표": rows,
           "gate_원문1-7": {"요구_k": list(KS_SPEC),
                            "그림_k": list(KS_FIG),
                            "k=100_실제_랭크": [r for r in rows if r["k"] == 100][0]["실제 랭크"],
                            "판정": ("k=10·50·100 을 그림 02 에 실었다. k=100 은 64×64 의 랭크 상한(64) 때문에 "
                                     "k=64 와 같은 그림이 되며, 그 사실을 제목에 적는다(숨기지 않는다)")},
           "조건": "64×64 float64 합성 이미지(사인파+가우시안 σ=12), 시드 42, np.linalg.svd full_matrices=False, "
                   "저장량 = k(m+n+1) / mn",
           "판정": "k≥64 는 랭크 상한이라 더 저장해도 개선 0 → 저장비율이 100%를 넘는 구간이 '압축 실패'로 보인다"}
    return write("svd.json", out)


if __name__ == "__main__":
    d = main()
    for r in d["표"]:
        print(r)
    print(d["gate_원문1-7"])
    print(d["판정"])
