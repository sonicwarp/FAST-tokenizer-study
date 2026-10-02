"""
step5_jpeg_quantization.py
===========================
DCT 공부 Step 5: JPEG 양자화(Quantization)

DCT 계수 → 양자화 테이블로 나누기 → 반올림 → 정수화
이것이 JPEG의 실제 손실 압축 단계
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# JPEG 표준 휘도(Y채널) 양자화 테이블 (quality=50 기준)
JPEG_Q_TABLE = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99],
], dtype=float)
# 좌상단(저주파) = 작은 값 → 세밀하게 표현
# 우하단(고주파) = 큰 값  → 거칠게 표현 (인간 눈이 덜 민감)

def dct2(b):
    return dct(dct(b, norm='ortho', axis=0), norm='ortho', axis=1)

def idct2(b):
    return idct(idct(b, norm='ortho', axis=0), norm='ortho', axis=1)

def jpeg_quality_table(Q_base, quality):
    """quality 파라미터(1~100)에 따라 양자화 테이블 스케일 조정"""
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    Q = np.floor((Q_base * scale + 50) / 100)
    return np.clip(Q, 1, 255)

# ── 예제 블록 ────────────────────────────────────────────────────────────────
np.random.seed(0)
block = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        block[i, j] = 128 + 40 * np.cos(np.pi * i / 7) + 25 * np.sin(np.pi * j / 7)
block += np.random.randn(8, 8) * 8

print("=== JPEG 양자화 단계 ===\n")
print("① DCT 적용:")
B = dct2(block)
print(np.round(B, 1))

print(f"\n② 양자화 테이블으로 나누고 반올림 (quality=50):")
Q = jpeg_quality_table(JPEG_Q_TABLE, 50)
B_quantized = np.round(B / Q)   # 나누고 반올림 → 정수
print(B_quantized.astype(int))
print("  → 고주파 계수들이 0으로 몰림 (양자화 테이블 값이 크므로 나누면 0)")

print(f"\n③ 0이 아닌 계수 수: {np.sum(B_quantized != 0)}/64  (나머지는 버려짐)")

print(f"\n④ 역양자화 + IDCT → 복원:")
B_dequantized = B_quantized * Q    # 역양자화 (원래 스케일로 복원)
block_rec = idct2(B_dequantized)
mse = np.mean((block - block_rec) ** 2)
print(f"   MSE = {mse:.2f}")

# ── quality 비교 ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle('JPEG 품질(quality) 설정에 따른 압축 비교')

axes[0, 0].imshow(block, cmap='gray', vmin=70, vmax=190)
axes[0, 0].set_title('원본')
axes[0, 0].axis('off')

axes[0, 1].imshow(Q, cmap='hot')
axes[0, 1].set_title('양자화 테이블 (quality=50)\n좌상단=세밀, 우하단=거칠게')
for i in range(8):
    for j in range(8):
        axes[0, 1].text(j, i, int(Q[i, j]), ha='center', va='center',
                        fontsize=6, color='white')

qualities = [10, 50, 90]
for idx, q in enumerate(qualities):
    Q_q = jpeg_quality_table(JPEG_Q_TABLE, q)
    B_q = np.round(B / Q_q) * Q_q
    rec = idct2(B_q)
    nonzero = np.sum(np.round(B / Q_q) != 0)
    mse_q = np.mean((block - rec) ** 2)

    row, col = divmod(idx + 3, 3)
    axes[row, col].imshow(rec, cmap='gray', vmin=70, vmax=190)
    axes[row, col].set_title(f'quality={q}\n비영 계수={nonzero}/64, MSE={mse_q:.1f}')
    axes[row, col].axis('off')
    print(f"quality={q:3d}: 비영 계수 {nonzero}/64, MSE={mse_q:.2f}")

plt.tight_layout()
plt.savefig('step5_jpeg_quantization.png', dpi=100)
plt.show()
print("→ step5_jpeg_quantization.png 저장됨")

print()
print("=== 핵심 정리 ===")
print("JPEG 압축 = DCT + 양자화 + 엔트로피 코딩(허프만)")
print("  DCT:   공간 → 주파수 변환 (무손실)")
print("  양자화: 계수를 정수로 반올림 (이 단계에서 손실 발생)")
print("  고주파 계수 → 테이블 값이 크다 → 나눠서 반올림하면 0")
print("  → 많은 계수가 0 → 이후 압축 효율 높음")
