"""
fast_step1_problem.py
======================
FAST 공부 Step 1: 왜 FAST가 필요한가?

문제: 로봇 제어 액션 시퀀스를 언어 모델 토큰으로 어떻게 표현하나?

OpenVLA 방식 (기존):
  - 매 타임스텝 액션값을 그냥 bin으로 나눠 토큰화
  - T=50 스텝, D=7차원 → 50×7 = 350개 토큰 필요
  - 토큰이 너무 많다 → 비효율

FAST 방식 (Physical Intelligence, 2024):
  - 액션 시퀀스 전체를 DCT로 압축 → 저주파 계수만 유지
  - T=50 스텝 → K≈10개 계수만으로 충분
  - K×7 = 70개 토큰 (약 5배 압축!)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import dct

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 1. 로봇 액션 시퀀스 설정 ──────────────────────────────────────────────────
T = 50    # 타임스텝 수 (50 Hz로 1초 = 50 스텝)
D = 7     # 액션 차원수: x, y, z, roll, pitch, yaw, gripper

# 차원 이름
DIM_NAMES = ['x 이동', 'y 이동', 'z 이동', 'roll', 'pitch', 'yaw', 'gripper']

print("=" * 60)
print("로봇 액션 시퀀스 설명")
print("=" * 60)
print(f"T = {T}  (타임스텝: 50Hz로 1초 동안의 제어 신호)")
print(f"D = {D}  (액션 차원: x,y,z 이동 + roll,pitch,yaw 회전 + gripper 개폐)")
print(f"액션 행렬 shape: [{T}, {D}]")
print()

# 실제처럼 보이는 액션 시퀀스 생성 (물체 집기 동작 시뮬레이션)
np.random.seed(42)
t = np.linspace(0, 1, T)

actions = np.zeros((T, D))
# x: 앞으로 이동 (부드러운 S자 곡선)
actions[:, 0] = 0.3 * (3 * t**2 - 2 * t**3) + 0.02 * np.sin(2 * np.pi * t * 3) * np.exp(-t * 2)
# y: 약간 옆으로
actions[:, 1] = 0.05 * np.sin(np.pi * t) + 0.01 * np.random.randn(T)
# z: 내려갔다가 올라옴
actions[:, 2] = -0.15 * np.sin(np.pi * t) + 0.02 * np.random.randn(T)
# roll, pitch, yaw: 작은 회전
actions[:, 3] = 0.1 * np.sin(2 * np.pi * t * 0.5) + 0.005 * np.random.randn(T)
actions[:, 4] = 0.05 * np.cos(np.pi * t) + 0.005 * np.random.randn(T)
actions[:, 5] = 0.02 * t + 0.005 * np.random.randn(T)
# gripper: 처음엔 열려있다가 닫힘
actions[:, 6] = np.where(t < 0.7, 0.8, 0.8 - (t - 0.7) / 0.3 * 0.7)

print("액션 시퀀스 첫 3 스텝:")
print(f"{'차원':>10}  {'t=0':>8}  {'t=1':>8}  {'t=2':>8}")
for d in range(D):
    print(f"{DIM_NAMES[d]:>10}  {actions[0,d]:+8.4f}  {actions[1,d]:+8.4f}  {actions[2,d]:+8.4f}")

# ── 2. OpenVLA 방식 vs FAST 방식 토큰 수 비교 ────────────────────────────────
print()
print("=" * 60)
print("토큰 수 비교")
print("=" * 60)

openvla_tokens = T * D
print(f"OpenVLA: 매 스텝 각 차원을 독립적으로 bin → {T}×{D} = {openvla_tokens}개 토큰")

# FAST: DCT 후 에너지 기준으로 K 결정
A_dct = dct(actions, norm='ortho', axis=0)   # shape=[T, D], axis=0 = 시간축
energies_per_dim = A_dct ** 2                 # 각 (k, d) 위치의 에너지

total_energy_per_dim = energies_per_dim.sum(axis=0)   # 차원별 총 에너지
cum_energy_ratio = np.cumsum(energies_per_dim, axis=0) / total_energy_per_dim   # 누적 에너지 비율

# 각 차원별로 에너지 95% 커버에 필요한 K
K_per_dim = [(cum_energy_ratio[:, d] >= 0.95).argmax() + 1 for d in range(D)]
K = max(K_per_dim)   # 모든 차원 커버하는 K

print(f"FAST:    DCT 후 저주파 K개 계수만 유지 → {K}×{D} = {K*D}개 토큰")
print(f"         (각 차원별 에너지 95% 커버 K: {K_per_dim})")
print(f"압축률: {openvla_tokens / (K*D):.1f}배 감소")

# ── 3. DCT 에너지 분포 시각화 ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle(f'DCT 에너지 분포 (각 액션 차원, T={T})')

for d in range(D):
    ax = axes[d // 4][d % 4]
    energy_d = energies_per_dim[:, d]
    cum_e = np.cumsum(energy_d) / total_energy_per_dim[d] * 100

    ax2 = ax.twinx()
    ax.bar(range(T), energy_d, alpha=0.6, color='steelblue', label='에너지')
    ax2.plot(range(T), cum_e, 'r-', linewidth=1.5, label='누적 %')
    ax2.axhline(95, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylim(0, 105)

    k95 = K_per_dim[d]
    ax.axvline(k95 - 0.5, color='red', linewidth=2, alpha=0.8)
    ax.set_title(f'{DIM_NAMES[d]}\n(K≥{k95}이면 95%)')
    ax.set_xlabel('주파수 k')
    ax.set_xlim(-0.5, 20)   # 처음 20개만 표시

# 마지막 칸: 토큰 수 비교
ax_last = axes[1][3]
methods = ['OpenVLA', 'FAST']
token_counts = [openvla_tokens, K * D]
colors = ['salmon', 'steelblue']
bars = ax_last.bar(methods, token_counts, color=colors)
for bar, count in zip(bars, token_counts):
    ax_last.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                 str(count), ha='center', va='bottom', fontweight='bold')
ax_last.set_ylabel('토큰 수')
ax_last.set_title(f'토큰 수 비교\n(FAST: {openvla_tokens//(K*D)}배 적음)')

plt.tight_layout()
plt.savefig('fast_step1_problem.png', dpi=100)
plt.show()
print("→ fast_step1_problem.png 저장됨")
