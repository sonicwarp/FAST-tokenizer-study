"""
fast_step4_full_pipeline.py
============================
FAST 공부 Step 4: 전체 파이프라인 통합

FAST VLA 전체 흐름:
  ① 카메라 이미지 → ViT → 이미지 토큰
  ② 언어 명령어 → 텍스트 토큰
  ③ 과거 액션 → FAST 토큰 (DCT + 양자화)
  ④ LLM이 세 종류 토큰을 함께 처리 → 다음 FAST 토큰 예측
  ⑤ 예측된 FAST 토큰 → IDCT → 실제 액션 복원 → 로봇 실행

이 파일에서는 ③⑤ (액션 토크나이징) 전체를 단계별로 출력하며 확인
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════
T = 50        # 타임스텝
D = 7         # 액션 차원 (x,y,z,roll,pitch,yaw,gripper)
K = 10        # FAST 계수 수
N_BINS = 256  # 양자화 bin 수

DIM_NAMES = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']

np.random.seed(42)
t = np.linspace(0, 1, T)

# 로봇 액션 시퀀스 (물체 집기 동작)
actions = np.zeros((T, D))
actions[:, 0] = 0.3 * (3*t**2 - 2*t**3) + 0.01 * np.random.randn(T)
actions[:, 1] = 0.08 * np.sin(np.pi * t) + 0.008 * np.random.randn(T)
actions[:, 2] = -0.12 * np.sin(np.pi * t) + 0.008 * np.random.randn(T)
actions[:, 3] = 0.08 * np.sin(2*np.pi*t*0.5) + 0.003 * np.random.randn(T)
actions[:, 4] = 0.06 * np.cos(np.pi * t) + 0.003 * np.random.randn(T)
actions[:, 5] = 0.015 * t + 0.003 * np.random.randn(T)
actions[:, 6] = np.where(t < 0.7, 0.8, 0.8 - (t-0.7)/0.3 * 0.7)

# ══════════════════════════════════════════════════════════════════════════════
# 전체 파이프라인: 단계별 shape 출력
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("FAST 전체 인코딩 파이프라인")
print("=" * 65)

# ① 입력
print(f"① 원본 액션 시퀀스:          shape={actions.shape}  [{T}타임스텝 × {D}차원]")

# ② DCT (axis=0 = 시간 방향)
dct_all = dct(actions, norm='ortho', axis=0)
print(f"② DCT 변환 (axis=0):         shape={dct_all.shape}  [주파수 T개 × {D}차원]")

# ③ 앞 K개 선택
top_k = dct_all[:K, :]
print(f"③ 저주파 K={K}개 선택:          shape={top_k.shape}  [{K}계수 × {D}차원]")

# ④ flatten
coeffs_flat = top_k.flatten()
print(f"④ flatten:                   shape={coeffs_flat.shape}  [K×D = {K*D}개 실수값]")

# ⑤ 양자화 범위 결정 (실제론 fit()에서 학습 데이터로 결정)
coeff_min = coeffs_flat.min() - 0.05  # 약간 여유
coeff_max = coeffs_flat.max() + 0.05
bin_width = (coeff_max - coeff_min) / N_BINS

# ⑥ 정규화 → bin 번호
norm = (coeffs_flat - coeff_min) / (coeff_max - coeff_min)
token_ids = np.clip(np.floor(norm * N_BINS).astype(int), 0, N_BINS - 1)
print(f"⑤ 양자화 (정규화 → bin):     shape={token_ids.shape}  [정수 토큰 ID, 0~{N_BINS-1}]")

print()
print(f"  토큰 수 비교:")
print(f"    OpenVLA 방식: T×D = {T}×{D} = {T*D}개 토큰")
print(f"    FAST 방식:    K×D = {K}×{D} = {K*D}개 토큰  ({T*D//(K*D)}배 압축)")
print()
print(f"  토큰 샘플 (처음 14개): {token_ids[:14]}")
print(f"  토큰 범위: [{token_ids.min()}, {token_ids.max()}]")

# ══════════════════════════════════════════════════════════════════════════════
# 디코딩 파이프라인
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("FAST 전체 디코딩 파이프라인")
print("=" * 65)

# ⑥ bin 중심값으로 복원
coeffs_rec = coeff_min + token_ids * bin_width + bin_width / 2
print(f"⑥ 역양자화 (bin→계수값):     shape={coeffs_rec.shape}")

# ⑦ reshape [K, D]
top_k_rec = coeffs_rec.reshape(K, D)
print(f"⑦ reshape:                   shape={top_k_rec.shape}  [{K} × {D}]")

# ⑧ 제로패딩 [T, D]
dct_padded = np.zeros((T, D))
dct_padded[:K, :] = top_k_rec
print(f"⑧ 제로패딩:                  shape={dct_padded.shape}  [{T} × {D}]")

# ⑨ IDCT
actions_rec = idct(dct_padded, norm='ortho', axis=0)
print(f"⑨ IDCT:                      shape={actions_rec.shape}  ← 원본과 동일!")

mse = np.mean((actions - actions_rec) ** 2)
print(f"\n최종 MSE: {mse:.6f}  (K={K}개 계수로 충분히 정확한 복원)")

# ══════════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f'FAST 전체 파이프라인 (T={T}, D={D}, K={K}, n_bins={N_BINS})', fontsize=13)

# 상단: 액션 복원 비교 (7차원)
for d in range(D):
    ax = fig.add_subplot(3, 4, d + 1)
    ax.plot(t, actions[:, d], 'b-', linewidth=1.5, label='원본', alpha=0.8)
    ax.plot(t, actions_rec[:, d], 'r--', linewidth=1.5, label='복원', alpha=0.8)
    mse_d = np.mean((actions[:, d] - actions_rec[:, d]) ** 2)
    ax.set_title(f'{DIM_NAMES[d]}  MSE={mse_d:.5f}', fontsize=9)
    if d == 0:
        ax.legend(fontsize=7)
    ax.set_xlabel('시간', fontsize=7)
    ax.tick_params(labelsize=7)

# 우하단: 토큰 수 비교 막대그래프
ax_bar = fig.add_subplot(3, 4, 8)
labels = ['OpenVLA\n(T×D)', 'FAST\n(K×D)']
counts = [T * D, K * D]
colors = ['#FF6B6B', '#4ECDC4']
bars = ax_bar.bar(labels, counts, color=colors, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts):
    ax_bar.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)
ax_bar.set_ylabel('토큰 수')
ax_bar.set_title(f'토큰 수 비교\n(FAST: {T*D//(K*D)}배 적음)')

# 하단: DCT 계수 분포
ax_dct = fig.add_subplot(3, 4, (9, 12))
for d in range(D):
    energy_d = dct_all[:, d] ** 2
    energy_d_norm = energy_d / energy_d.sum() * 100
    ax_dct.plot(range(min(30, T)), energy_d_norm[:30],
                marker='.', markersize=4, label=DIM_NAMES[d], alpha=0.7)

ax_dct.axvline(K - 0.5, color='red', linewidth=2, linestyle='--', label=f'K={K} 경계')
ax_dct.set_xlabel('주파수 k')
ax_dct.set_ylabel('에너지 비율 (%)')
ax_dct.set_title(f'DCT 에너지 분포 (k < {K} 구간에 에너지 집중)')
ax_dct.legend(fontsize=7, ncol=2)
ax_dct.set_xlim(-0.5, 29.5)

plt.tight_layout()
plt.savefig('fast_step4_full_pipeline.png', dpi=100)
plt.show()
print("→ fast_step4_full_pipeline.png 저장됨")

# ══════════════════════════════════════════════════════════════════════════════
# 최종 요약
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("FAST vs OpenVLA 최종 요약")
print("=" * 65)
print()
print("OpenVLA:  각 타임스텝 액션값 → 바로 bin → 토큰")
print("          [50, 7] → 350개 토큰 (단순하지만 많음)")
print()
print("FAST:     액션 시퀀스 전체 → DCT → 저주파 K개 → bin → 토큰")
print(f"          [50, 7] → DCT → [:10, :] → 70개 토큰 (5배 압축)")
print()
print("FAST의 핵심 가정:")
print("  '로봇 액션은 부드럽게 변한다 = 저주파에 에너지 집중'")
print("  → 저주파 계수 K개만으로 시퀀스 전체를 잘 표현 가능")
print()
print("장점:")
print("  - 토큰 수 감소 → LLM 입력 길이 단축 → 속도/비용 개선")
print("  - 전체 시퀀스를 한 번에 예측 → 일관성 향상")
print("  - 기존 언어 모델 어휘(vocab)에 액션 토큰 직접 추가 가능")
