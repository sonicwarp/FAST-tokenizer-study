"""
step1_cosine_basis.py
======================
DCT 공부 Step 1: 코사인 기저(basis) 함수란 무엇인가?

핵심 개념:
  - 기저(basis) : 어떤 신호도 이 기저들의 합으로 표현할 수 있는 "기본 블록"
  - DCT 기저   : φ_k[n] = cos( π·k·(2n+1) / 2N )
  - 직교성     : 서로 다른 두 기저를 내적하면 0  → 서로 완전히 독립
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

N = 8   # 신호 길이 (샘플 8개)

# ── 1. 기저 함수 정의 ─────────────────────────────────────────────────────────
# φ_k[n] = cos( π·k·(2n+1) / 2N )
#   k : 주파수 번호 (0 = DC, 1 = 가장 느린 진동, N-1 = 가장 빠른 진동)
#   n : 샘플 위치  (0 ~ N-1)
n = np.arange(N)                          # [0, 1, 2, 3, 4, 5, 6, 7]

def phi(k, n, N):
    """k번째 DCT 코사인 기저 함수값 계산"""
    return np.cos(np.pi * k * (2 * n + 1) / (2 * N))

# ── 2. 기저 함수 시각화 ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 5))
fig.suptitle(f'DCT 기저 함수 φ_k[n]  (N={N})', fontsize=13)

for k in range(N):
    ax = axes[k // 4][k % 4]
    basis = phi(k, n, N)
    ax.stem(n, basis, basefmt='k-')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_title(f'φ_{k}[n]  (k={k})')
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel('n (샘플 위치)')

plt.tight_layout()
plt.savefig('step1_basis.png', dpi=100)
plt.show()
print("→ step1_basis.png 저장됨")

# ── 3. 직교성 확인 ────────────────────────────────────────────────────────────
# 직교성: 내적 = Σ φ_i[n] · φ_j[n] = 0  (i ≠ j 일 때)
# 이 성질 때문에 각 주파수 성분은 서로 '독립'으로 다룰 수 있다.
print("\n=== 기저 함수 내적 표 (직교성 검증) ===")
print(f"{'':4}", end='')
for j in range(N):
    print(f"  φ_{j}", end='')
print()
for i in range(N):
    print(f"φ_{i} ", end='')
    for j in range(N):
        dot = np.dot(phi(i, n, N), phi(j, n, N))
        print(f"  {dot:4.1f}", end='')
    print()

print()
print("→ 대각선(같은 기저끼리): 0이 아닌 값 (자기 자신과의 유사도)")
print("→ 비대각선(다른 기저끼리): 0에 가까운 값 (완전히 독립)")

# ── 4. 임의 신호를 기저로 분해 ──────────────────────────────────────────────
# 신호 x = a0·φ_0 + a1·φ_1 + ... + a_{N-1}·φ_{N-1}
# 계수  a_k = Σ x[n] · φ_k[n] / Σ φ_k[n]²
x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])   # 임의 신호
print(f"\n분석할 신호: {x}")

print("\n각 기저와의 내적(유사도) 계산:")
coeffs = []
for k in range(N):
    basis_k = phi(k, n, N)
    dot = np.dot(x, basis_k)              # 신호가 k번 기저와 얼마나 닮았나
    norm_sq = np.dot(basis_k, basis_k)    # 기저 자신의 크기 제곱
    a_k = dot / norm_sq                   # 정규화된 계수
    coeffs.append(a_k)
    print(f"  k={k}: 내적={dot:7.3f}, 기저크기²={norm_sq:.3f}, 계수 a_{k}={a_k:.3f}")

# 재구성
x_reconstructed = sum(coeffs[k] * phi(k, n, N) for k in range(N))
print(f"\n원본 신호:    {np.round(x, 3)}")
print(f"재구성 신호:  {np.round(x_reconstructed, 3)}")
print(f"오차 (max):   {np.max(np.abs(x - x_reconstructed)):.2e}  ← 거의 0 (완벽 재구성)")
