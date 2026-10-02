"""
step2_dct1d_numpy.py
=====================
DCT 공부 Step 2: DCT 변환을 행렬곱으로 이해하기

핵심 흐름:
  1. DCT 행렬 D = "코사인 파형표"  (각 행 = 하나의 기저 파형)
  2. X = D @ x   → 각 기저와의 유사도(내적) 계산
  3. X[k]² = 에너지  → 부호 무관 크기 비교
  4. D.T @ X = x  → 역변환 (완벽 재구성)
  5. 고주파 계수 버리기 → 압축 (약간의 손실)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


# ══════════════════════════════════════════════════════════════════════════════
# Part 0. 손으로 계산하는 가장 간단한 예제 (N=4)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Part 0. N=4 예제: 내적을 손으로 계산해보기")
print("=" * 60)

# 신호: 길이 4인 간단한 값
x4 = np.array([8.0, 6.0, 4.0, 2.0])
N4 = 4
n4 = np.arange(N4)    # [0, 1, 2, 3]

print(f"신호 x = {x4}")
print()

# k=0 기저: cos( π·0·(2n+1) / 8 ) = cos(0) = 1 for all n
# → 완전히 평평한 파형 (DC 성분)
phi0 = np.cos(np.pi * 0 * (2 * n4 + 1) / (2 * N4))   # [1, 1, 1, 1]
print(f"k=0 기저 φ_0 = {phi0}  ← 모두 1 (평평)")
dot0 = np.dot(x4, phi0)
print(f"  내적 = {x4[0]}×{phi0[0]:.0f} + {x4[1]}×{phi0[1]:.0f} + "
      f"{x4[2]}×{phi0[2]:.0f} + {x4[3]}×{phi0[3]:.0f} = {dot0:.1f}")
print(f"  → X[0] = {dot0:.3f}  (신호 전체 평균과 비례)")
print()

# k=1 기저: 한 번 출렁이는 파형
phi1 = np.cos(np.pi * 1 * (2 * n4 + 1) / (2 * N4))
print(f"k=1 기저 φ_1 = {np.round(phi1, 3)}  ← 한 번 출렁")
dot1 = np.dot(x4, phi1)
print(f"  내적 = {' + '.join(f'{x4[i]:.0f}×{phi1[i]:.3f}' for i in range(N4))} = {dot1:.3f}")
print(f"  → X[1] = {dot1:.3f}")
print()

# k=2, k=3도 마찬가지
phi2 = np.cos(np.pi * 2 * (2 * n4 + 1) / (2 * N4))
phi3 = np.cos(np.pi * 3 * (2 * n4 + 1) / (2 * N4))
dot2, dot3 = np.dot(x4, phi2), np.dot(x4, phi3)
print(f"k=2 기저 φ_2 = {np.round(phi2, 3)}, 내적 X[2] = {dot2:.3f}")
print(f"k=3 기저 φ_3 = {np.round(phi3, 3)}, 내적 X[3] = {dot3:.3f}")
print()
print("결론: 신호 x가 k=0(평평한) 파형과 가장 많이 닮았다 → X[0]이 가장 크다")


# ══════════════════════════════════════════════════════════════════════════════
# Part 1. DCT 행렬 = "코사인 파형표"
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Part 1. DCT 행렬 만들기")
print("=" * 60)

def make_dct_matrix_normalized(N):
    """
    N×N DCT 행렬 D를 만든다.
    D[k, n] = φ_k[n] = cos( π·k·(2n+1) / 2N )
    각 행(row) = 하나의 기저 파형
    정규화: k=0 행 × sqrt(1/N),  k≥1 행 × sqrt(2/N)
    """
    k = np.arange(N)                          # [0, 1, ..., N-1]  (행 번호 = 주파수)
    n = np.arange(N).reshape(-1, 1)           # [[0],[1],...,[N-1]]  (열 번호 = 샘플 위치)
    # cos( π·k·(2n+1) / 2N ) 계산 → 결과 shape=(N,N), 행=샘플, 열=주파수
    # 전치(.T)하면 행=주파수, 열=샘플
    D = np.cos(np.pi * k * (2 * n + 1) / (2 * N)).T
    # 정규화 (ortho 모드와 동일)
    D[0, :] *= np.sqrt(1 / N)    # k=0: DC 행
    D[1:, :] *= np.sqrt(2 / N)   # k≥1: AC 행
    return D

N = 8
x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])

D = make_dct_matrix_normalized(N)
print(f"DCT 행렬 D  (shape={D.shape})")
print("  - 행(row) 0 = k=0 기저 (평평한 파형)")
print("  - 행(row) 1 = k=1 기저 (가장 느린 진동)")
print("  - 행(row) 7 = k=7 기저 (가장 빠른 진동)")
print(np.round(D, 3))


# ══════════════════════════════════════════════════════════════════════════════
# Part 2. X = D @ x : 각 기저와의 유사도 계산
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Part 2. DCT 계수 계산: X = D @ x")
print("=" * 60)

X = D @ x    # shape=(N,) : 각 주파수 성분의 계수

print(f"신호 x = {x}")
print(f"DCT 계수 X = D @ x = {np.round(X, 3)}")
print()
print("각 계수의 의미:")
for k in range(N):
    energy = X[k] ** 2   # 에너지 = 계수²
    print(f"  X[{k}] = {X[k]:+8.3f}  (에너지={energy:7.3f}) "
          f"← 신호가 k={k} 파형과 '{energy:.1f}만큼' 닮음")

print()
# scipy dct로 검증
X_scipy = dct(x, norm='ortho')
print(f"scipy dct 결과:  {np.round(X_scipy, 3)}")
print(f"우리 D@x 결과:   {np.round(X, 3)}")
print(f"차이(최대):      {np.max(np.abs(X - X_scipy)):.2e}  ← 동일!")


# ══════════════════════════════════════════════════════════════════════════════
# Part 3. 에너지 = 계수² : 왜 제곱하나?
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Part 3. 에너지 = 계수²  (왜 제곱?)")
print("=" * 60)
print()
print("계수가 +3 이든 -3 이든 → '3만큼 닮았다'는 크기는 같다")
print("  계수 X[k]    = +3  → 에너지 = 9")
print("  계수 X[k]    = -3  → 에너지 = 9")
print("  절댓값 |X[k]| 도 되지만, 계산 편의상 제곱을 주로 쓴다")
print()

energies = X ** 2
total_energy = np.sum(energies)
print(f"전체 에너지: {total_energy:.2f}")
for k in range(N):
    ratio = energies[k] / total_energy * 100
    bar = '█' * int(ratio / 2)
    print(f"  k={k}: 에너지={energies[k]:7.2f}  ({ratio:5.1f}%)  {bar}")

print()
cumsum = np.cumsum(np.sort(energies)[::-1]) / total_energy * 100
# 실제로는 k=0,1,2,...순서대로 누적
cum_ordered = np.cumsum(energies) / total_energy * 100
for k in range(N):
    print(f"  k=0~{k} 누적 에너지: {cum_ordered[k]:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Part 4. 역변환: x = D.T @ X  (완벽 재구성)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Part 4. 역변환: x = D.T @ X")
print("=" * 60)

x_reconstructed = D.T @ X   # D는 직교행렬 → D.T = D^{-1}
print(f"원본 신호:    {np.round(x, 4)}")
print(f"재구성 신호:  {np.round(x_reconstructed, 4)}")
print(f"최대 오차:    {np.max(np.abs(x - x_reconstructed)):.2e}  ← DCT 자체는 무손실!")


# ══════════════════════════════════════════════════════════════════════════════
# Part 5. 압축: 고주파 계수를 버리면?
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Part 5. 압축: 고주파 계수를 버린다")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('DCT 압축: 유지하는 계수 개수에 따른 재구성 품질')

for idx, K in enumerate([2, 4, 8]):
    X_compressed = X.copy()
    X_compressed[K:] = 0          # k≥K인 고주파 계수를 0으로 → 버림

    x_rec = D.T @ X_compressed    # 역변환

    ax = axes[idx]
    ax.plot(x, 'o-', label='원본', markersize=6)
    ax.plot(x_rec, 's--', label=f'재구성(K={K})', markersize=6)
    ax.set_title(f'K={K}개 계수 사용')
    ax.legend()
    ax.set_xlabel('샘플 위치 n')
    err = np.mean((x - x_rec) ** 2)
    ax.set_ylabel(f'MSE={err:.3f}')

    print(f"K={K}: 계수 {K}개만 사용 → MSE={err:.4f}")

plt.tight_layout()
plt.savefig('step2_compression.png', dpi=100)
plt.show()
print("→ step2_compression.png 저장됨")
print()
print("정리: DCT 자체는 무손실. 손실은 계수를 '의도적으로 버릴 때' 발생.")
print("      저주파(낮은 k) 계수에 에너지 집중 → 조금만 버려도 거의 손실 없음")
