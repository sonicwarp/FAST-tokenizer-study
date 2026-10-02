"""
step4_dct2d_image.py
=====================
DCT 공부 Step 4: 2D DCT - 이미지에 적용

JPEG는 이미지를 8×8 블록으로 나누고 각 블록에 2D DCT 적용
2D DCT = 먼저 행(가로)에 1D DCT, 그 다음 열(세로)에 1D DCT
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 1. 2D DCT 함수 ────────────────────────────────────────────────────────────
def dct2(block):
    """2D DCT-II (axis=0: 세로, axis=1: 가로 순서로 적용)"""
    return dct(dct(block, norm='ortho', axis=0), norm='ortho', axis=1)

def idct2(block):
    """2D IDCT"""
    return idct(idct(block, norm='ortho', axis=0), norm='ortho', axis=1)

# ── 2. 8×8 예제 블록 ──────────────────────────────────────────────────────────
np.random.seed(42)
# 실제 이미지처럼 천천히 변하는 신호 (부드러운 그라디언트 + 약간의 노이즈)
block = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        block[i, j] = 128 + 30 * np.cos(np.pi * i / 7) + 20 * np.sin(np.pi * j / 7)
block += np.random.randn(8, 8) * 5   # 약간의 노이즈 추가

print("=== 원본 8×8 블록 (픽셀값) ===")
print(np.round(block, 1))

# ── 3. 2D DCT 적용 ────────────────────────────────────────────────────────────
B = dct2(block)
print("\n=== 2D DCT 계수 ===")
print(np.round(B, 1))
print(f"\n(0,0) 위치 = DC 계수 = {B[0,0]:.1f}  ← 블록 전체 평균과 비례")
print(f"에너지의 {np.sum(B[:2, :2]**2) / np.sum(B**2) * 100:.1f}%가 좌상단 2×2에 집중")

# ── 4. 압축: 좌상단 K×K만 유지 ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle('2D DCT 압축 - 유지하는 계수 수에 따른 품질')

axes[0, 0].imshow(block, cmap='gray', vmin=80, vmax=180)
axes[0, 0].set_title('원본 블록')
axes[0, 0].axis('off')

axes[0, 1].imshow(np.abs(B), cmap='hot')
axes[0, 1].set_title('DCT 계수 (절댓값)')
axes[0, 1].set_xlabel('가로 주파수')
axes[0, 1].set_ylabel('세로 주파수')

for idx, K in enumerate([2, 4, 8]):
    B_compressed = np.zeros_like(B)
    B_compressed[:K, :K] = B[:K, :K]   # 좌상단 K×K만 유지
    block_rec = idct2(B_compressed)

    coeff_count = K * K
    total_count = 8 * 8
    mse = np.mean((block - block_rec) ** 2)

    if idx < 2:
        ax = axes[1, idx]
    else:
        ax = axes[0, 2]

    ax.imshow(block_rec, cmap='gray', vmin=80, vmax=180)
    ax.set_title(f'{K}×{K}={coeff_count}개 계수\nMSE={mse:.1f}')
    ax.axis('off')
    print(f"K={K}: {coeff_count}/{total_count}개 계수 사용, MSE={mse:.2f}")

axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('step4_dct2d_image.png', dpi=100)
plt.show()
print("→ step4_dct2d_image.png 저장됨")

print()
print("=== 결론 ===")
print("2D DCT도 1D DCT와 같은 원리:")
print("  - 자연 이미지는 저주파 성분에 에너지 집중")
print("  - 좌상단(저주파) 계수만 유지해도 화질 거의 유지")
print("  - JPEG는 이 원리로 압축")
