"""
step3_dct_vs_dft.py
====================
DCT 공부 Step 3: DCT vs DFT 비교

핵심 개념:
  - DFT : 복소수 기저 (sin + cos 혼합) → 경계 불연속 문제
  - DCT : 순수 코사인 기저 → 경계가 자연스럽게 연속 → 에너지 집중 더 우수
  - DCT-II : 가장 많이 쓰이는 형태 (JPEG, MP3, FAST 모두 이걸 사용)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import dct, fft

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

N = 16
n = np.arange(N)

# ── 1. 경계 불연속 문제 ───────────────────────────────────────────────────────
# DFT는 신호가 '무한히 반복된다'고 가정 → 끝과 시작이 연결될 때 불연속 발생
# DCT는 신호를 '거울 반사(mirror)'하여 연장 → 경계가 항상 연속
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
              8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])  # 대칭 신호

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle('DCT vs DFT 비교')

# DFT 결과
X_dft = np.abs(fft(x))
axes[0, 0].stem(n, X_dft[:N//2 + 1])
axes[0, 0].set_title('DFT 스펙트럼 (대칭 신호)')
axes[0, 0].set_xlabel('주파수 빈 k')
axes[0, 0].set_ylabel('|X[k]|')

# DCT 결과
X_dct = dct(x, norm='ortho')
axes[0, 1].stem(n, np.abs(X_dct))
axes[0, 1].set_title('DCT 스펙트럼 (대칭 신호)')
axes[0, 1].set_xlabel('주파수 빈 k')
axes[0, 1].set_ylabel('|X[k]|')

# 비대칭 신호 - DFT의 경계 불연속 문제 확인
x2 = np.sin(2 * np.pi * np.arange(N) / N * 1.5) + 0.5
X_dft2 = np.abs(fft(x2))
X_dct2 = dct(x2, norm='ortho')

axes[1, 0].stem(np.arange(N//2 + 1), X_dft2[:N//2 + 1])
axes[1, 0].set_title('DFT 스펙트럼 (비대칭 신호) - 에너지 분산')
axes[1, 0].set_xlabel('주파수 빈 k')
axes[1, 0].set_ylabel('|X[k]|')

axes[1, 1].stem(n, np.abs(X_dct2))
axes[1, 1].set_title('DCT 스펙트럼 (비대칭 신호) - 에너지 집중')
axes[1, 1].set_xlabel('주파수 빈 k')
axes[1, 1].set_ylabel('|X[k]|')

plt.tight_layout()
plt.savefig('step3_dct_vs_dft.png', dpi=100)
plt.show()
print("→ step3_dct_vs_dft.png 저장됨")

# ── 2. 에너지 집중 비교 ───────────────────────────────────────────────────────
print("\n=== 에너지 집중 비교 (처음 4개 계수로 몇 %?) ===")
energy_dft = X_dft2 ** 2
energy_dct = X_dct2 ** 2
print(f"DFT: 처음 4개 계수 에너지 = {np.sum(energy_dft[:4]) / np.sum(energy_dft) * 100:.1f}%")
print(f"DCT: 처음 4개 계수 에너지 = {np.sum(energy_dct[:4]) / np.sum(energy_dct) * 100:.1f}%")
print("→ DCT가 더 적은 계수로 더 많은 에너지를 설명 → 압축에 유리")

# ── 3. 4가지 DCT 타입 ────────────────────────────────────────────────────────
print("\n=== DCT 타입 설명 ===")
print("DCT-I  : 양 끝점 포함 대칭  (잘 안 씀)")
print("DCT-II : 반개(half-sample) 대칭  ← JPEG, MP3, FAST가 쓰는 표준")
print("         φ_k[n] = cos( π·k·(2n+1)/2N )")
print("DCT-III: DCT-II의 역변환")
print("DCT-IV : 입력·출력 모두 half-sample 대칭 (일부 코덱)")
print()
print("우리가 공부한 것은 DCT-II (scipy.fft.dct의 기본값)")
