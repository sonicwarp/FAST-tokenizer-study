"""
fast_step2_action_dct.py
=========================
FAST 공부 Step 2: 액션 시퀀스에 DCT 적용하기

핵심: axis=0 (시간축)으로 DCT 적용
  - 액션 shape: [T, D] = [타임스텝, 차원]
  - axis=0 = 시간 방향으로 DCT → 각 차원(열)을 독립적으로 1D DCT

FAST 인코딩:
  dct_coeffs = scipy.fft.dct(actions, norm='ortho', axis=0)  # [T, D]
  fast_code   = dct_coeffs[:K, :]                             # [K, D] (앞 K개만)

FAST 디코딩:
  padded       = np.zeros([T, D])
  padded[:K,:] = fast_code               # K개 계수를 앞에 채우고
  actions_rec  = scipy.fft.idct(padded, norm='ortho', axis=0)  # 복원
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 0. 예제 액션 생성 ──────────────────────────────────────────────────────────
T, D = 50, 7
DIM_NAMES = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
np.random.seed(7)
t = np.linspace(0, 1, T)

actions = np.zeros((T, D))
actions[:, 0] = 0.3 * (3*t**2 - 2*t**3) + 0.01 * np.random.randn(T)
actions[:, 1] = 0.08 * np.sin(np.pi * t) + 0.008 * np.random.randn(T)
actions[:, 2] = -0.12 * np.sin(np.pi * t) + 0.008 * np.random.randn(T)
actions[:, 3] = 0.08 * np.sin(2*np.pi*t*0.5) + 0.003 * np.random.randn(T)
actions[:, 4] = 0.06 * np.cos(np.pi * t) + 0.003 * np.random.randn(T)
actions[:, 5] = 0.015 * t + 0.003 * np.random.randn(T)
actions[:, 6] = np.where(t < 0.7, 0.8, 0.8 - (t - 0.7) / 0.3 * 0.7)

print(f"원본 액션 shape: {actions.shape}  ← [T, D] = [{T}, {D}]")
print()

# ── 1. axis=0 DCT 의미 설명 ───────────────────────────────────────────────────
print("=" * 60)
print("axis=0 DCT: 시간 방향으로 각 차원을 독립적으로 변환")
print("=" * 60)
print()
print("actions[:, 0] = x차원 시계열 → 1D DCT 적용")
print("actions[:, 1] = y차원 시계열 → 1D DCT 적용")
print("...  (모든 차원에 독립적으로 동일한 DCT)")
print()
print("즉, axis=0 DCT = D개의 1D DCT를 병렬로 수행")

# ── 2. DCT 적용 ───────────────────────────────────────────────────────────────
dct_coeffs = dct(actions, norm='ortho', axis=0)   # shape: [T, D]
print(f"\nDCT 후 shape: {dct_coeffs.shape}  ← [T, D] 그대로")
print("(각 위치 dct_coeffs[k, d] = d번 차원 시계열의 k번 주파수 계수)")
print()

# ── 3. 에너지 집중 확인 ───────────────────────────────────────────────────────
print("각 차원의 에너지 집중도 (낮은 k에 에너지가 몰리는가?):")
for d in range(D):
    energy_d = dct_coeffs[:, d] ** 2
    total = energy_d.sum()
    k90 = (np.cumsum(energy_d) / total >= 0.90).argmax() + 1
    k95 = (np.cumsum(energy_d) / total >= 0.95).argmax() + 1
    print(f"  {DIM_NAMES[d]:>7}: 90%={k90:2d}개, 95%={k95:2d}개 계수로 커버")

# ── 4. FAST 인코딩/디코딩 함수 ──────────────────────────────────────────────
def fast_encode(actions, K):
    """
    액션 시퀀스 [T, D] → DCT → 앞 K개 유지 → [K, D]
    """
    dct_all = dct(actions, norm='ortho', axis=0)   # [T, D]
    return dct_all[:K, :]                           # [K, D]

def fast_decode(fast_code, T):
    """
    FAST 코드 [K, D] → 제로패딩 → [T, D] → IDCT → 원본 복원
    """
    K, D = fast_code.shape
    padded = np.zeros((T, D))          # [T, D] 크기 영행렬
    padded[:K, :] = fast_code          # 앞 K행에 계수 채우기
    return idct(padded, norm='ortho', axis=0)   # [T, D] 복원

# ── 5. K 값에 따른 복원 품질 비교 ────────────────────────────────────────────
print()
print("=" * 60)
print("K 값에 따른 복원 품질 (MSE)")
print("=" * 60)

K_values = [5, 10, 20, 50]
for K in K_values:
    code = fast_encode(actions, K)
    actions_rec = fast_decode(code, T)
    mse = np.mean((actions - actions_rec) ** 2)
    token_count = K * D
    print(f"  K={K:2d}: 토큰={token_count:3d}개, MSE={mse:.6f}")

# ── 6. 시각화: 원본 vs 복원 비교 ─────────────────────────────────────────────
K = 10
code = fast_encode(actions, K)
actions_rec = fast_decode(code, T)

fig, axes = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle(f'FAST 인코딩/디코딩 (K={K}, T={T})\n'
             f'토큰: {T*D}개 → {K*D}개 ({T*D/(K*D):.1f}x 압축)')

for d in range(D):
    ax = axes[d // 4][d % 4]
    ax.plot(t, actions[:, d], 'b-', linewidth=1.5, label='원본')
    ax.plot(t, actions_rec[:, d], 'r--', linewidth=1.5, label=f'복원(K={K})')
    mse_d = np.mean((actions[:, d] - actions_rec[:, d]) ** 2)
    ax.set_title(f'{DIM_NAMES[d]}\nMSE={mse_d:.5f}')
    ax.set_xlabel('시간')
    ax.legend(fontsize=7)

# 마지막: K별 MSE 곡선
ax_last = axes[1][3]
K_range = range(1, 31)
mses = []
for k in K_range:
    c = fast_encode(actions, k)
    r = fast_decode(c, T)
    mses.append(np.mean((actions - r)**2))

ax_last.semilogy(K_range, mses, 'b-o', markersize=3)
ax_last.axvline(10, color='red', linestyle='--', label='K=10')
ax_last.set_xlabel('K (유지 계수 수)')
ax_last.set_ylabel('MSE (log 스케일)')
ax_last.set_title('K vs 복원 오차')
ax_last.legend()
ax_last.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fast_step2_action_dct.png', dpi=100)
plt.show()
print("→ fast_step2_action_dct.png 저장됨")
print()
print("핵심:")
print(f"  K=10만으로도 MSE 충분히 작음 → {T*D}개 토큰이 {K*D}개로 압축")
print("  FAST의 핵심 아이디어: 로봇 액션은 부드럽다 → 저주파에 에너지 집중")
