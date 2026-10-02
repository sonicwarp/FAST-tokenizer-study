"""
fast_step3_tokenization.py
===========================
FAST 공부 Step 3: DCT 계수를 정수 토큰으로 변환

연속값(실수) DCT 계수 → 이산값(정수) 토큰 ID

파이프라인:
  계수값 → [min, max] 범위 확인 → n_bins 등분 → 해당 bin 번호 = 토큰 ID

FASTTokenizer:
  fit(training_actions)  : 학습 데이터로 계수 범위 [min, max] 파악
  encode(actions)        : 액션 시퀀스 → 정수 토큰 ID 목록
  decode(token_ids)      : 정수 토큰 ID → 액션 시퀀스 복원
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


class FASTTokenizer:
    def __init__(self, K=10, n_bins=256):
        """
        K       : 유지할 DCT 계수 수 (저주파부터 K개)
        n_bins  : 양자화 단계 수 (256 = 1바이트 = 언어모델 어휘 크기와 매핑)
        """
        self.K = K
        self.n_bins = n_bins
        self.coeff_min = None   # fit()에서 설정
        self.coeff_max = None

    def fit(self, training_actions_list):
        """
        학습 데이터 액션 목록을 받아 DCT 계수 범위를 파악한다.
        모든 에피소드의 계수값을 모아서 min/max를 구함.
        """
        all_coeffs = []
        for actions in training_actions_list:
            # [T, D] → DCT → [T, D] → [:K] → [K, D] → flatten → [K*D]
            dct_all = dct(actions, norm='ortho', axis=0)
            coeffs_flat = dct_all[:self.K, :].flatten()
            all_coeffs.append(coeffs_flat)

        all_coeffs = np.concatenate(all_coeffs)
        self.coeff_min = all_coeffs.min()
        self.coeff_max = all_coeffs.max()
        print(f"fit 완료: 계수 범위 [{self.coeff_min:.4f}, {self.coeff_max:.4f}]")

    def encode(self, actions, verbose=False):
        """
        액션 [T, D] → 정수 토큰 ID 목록 [K*D]

        단계:
          ① DCT 적용: [T, D] → [T, D]
          ② 앞 K개 선택: [T, D] → [K, D]
          ③ flatten: [K, D] → [K*D]
          ④ 정규화: [0, 1] 범위로 변환
          ⑤ floor×n_bins: 정수 bin 번호
          ⑥ clip: [0, n_bins-1] 범위 보장
        """
        T, D = actions.shape

        # ① DCT
        dct_all = dct(actions, norm='ortho', axis=0)   # [T, D]

        # ② 앞 K개
        top_k = dct_all[:self.K, :]   # [K, D]

        # ③ flatten
        coeffs = top_k.flatten()      # [K*D]

        # ④ 정규화: (계수 - min) / (max - min) → [0, 1]
        norm = (coeffs - self.coeff_min) / (self.coeff_max - self.coeff_min)

        # ⑤ bin 번호: floor(norm × n_bins)
        token_ids = np.floor(norm * self.n_bins).astype(int)

        # ⑥ clip: 혹시 범위를 벗어나면 클램핑
        token_ids = np.clip(token_ids, 0, self.n_bins - 1)

        if verbose:
            bin_width = (self.coeff_max - self.coeff_min) / self.n_bins
            print(f"\n인코딩 상세 (처음 10개 계수):")
            print(f"{'k':>3}  {'계수값':>10}  {'정규화':>8}  {'토큰ID':>6}  {'bin 범위'}")
            print("-" * 55)
            for i in range(min(10, len(coeffs))):
                d = i % D
                k = i // D
                lo = self.coeff_min + token_ids[i] * bin_width
                hi = lo + bin_width
                print(f"{k:>3}  {coeffs[i]:+10.4f}  {norm[i]:8.4f}  {token_ids[i]:>6d}"
                      f"  [{lo:.3f}, {hi:.3f})")

        return token_ids   # shape: [K*D]

    def decode(self, token_ids, T, D):
        """
        정수 토큰 ID [K*D] → 액션 [T, D]

        단계:
          ① bin 중심값으로 역변환
          ② [K, D]로 reshape
          ③ 제로패딩: [K, D] → [T, D]
          ④ IDCT
        """
        K, n_bins = self.K, self.n_bins
        bin_width = (self.coeff_max - self.coeff_min) / n_bins

        # ① 각 bin ID의 중심값으로 복원
        coeffs = self.coeff_min + token_ids * bin_width + bin_width / 2

        # ② reshape
        top_k = coeffs.reshape(K, D)   # [K, D]

        # ③ 제로패딩
        dct_padded = np.zeros((T, D))
        dct_padded[:K, :] = top_k      # 앞 K행에 계수 채우기

        # ④ IDCT
        actions_rec = idct(dct_padded, norm='ortho', axis=0)   # [T, D]
        return actions_rec


# ══════════════════════════════════════════════════════════════════════════════
# 예제 실행
# ══════════════════════════════════════════════════════════════════════════════
T, D = 50, 7
DIM_NAMES = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
np.random.seed(42)
t = np.linspace(0, 1, T)

def make_action_episode(seed):
    """임의 액션 에피소드 생성"""
    rng = np.random.RandomState(seed)
    actions = np.zeros((T, D))
    actions[:, 0] = 0.3 * (3*t**2 - 2*t**3) + 0.01 * rng.randn(T)
    actions[:, 1] = 0.08 * np.sin(np.pi * t) + 0.008 * rng.randn(T)
    actions[:, 2] = -0.12 * np.sin(np.pi * t) + 0.008 * rng.randn(T)
    actions[:, 3] = 0.08 * np.sin(2*np.pi*t*0.5) + 0.003 * rng.randn(T)
    actions[:, 4] = 0.06 * np.cos(np.pi * t) + 0.003 * rng.randn(T)
    actions[:, 5] = 0.015 * t + 0.003 * rng.randn(T)
    actions[:, 6] = np.where(t < 0.7, 0.8, 0.8 - (t - 0.7) / 0.3 * 0.7)
    return actions

# 학습 데이터 여러 에피소드
training_data = [make_action_episode(i) for i in range(20)]
test_actions = make_action_episode(99)

# ── 토크나이저 초기화 및 학습 ─────────────────────────────────────────────────
print("=" * 60)
tokenizer = FASTTokenizer(K=10, n_bins=256)
tokenizer.fit(training_data)

# ── 인코딩 ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("인코딩: 액션 [50, 7] → 토큰 ID [70]")
print("=" * 60)
token_ids = tokenizer.encode(test_actions, verbose=True)

print(f"\n토큰 ID (처음 20개): {token_ids[:20]}")
print(f"토큰 ID shape: {token_ids.shape}  ← K×D = {tokenizer.K}×{D} = {len(token_ids)}개")
print(f"토큰 범위: [{token_ids.min()}, {token_ids.max()}]  (0~255 사이)")

# ── 디코딩 ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("디코딩: 토큰 ID [70] → 액션 [50, 7]")
print("=" * 60)
actions_rec = tokenizer.decode(token_ids, T=T, D=D)
mse = np.mean((test_actions - actions_rec) ** 2)
print(f"복원 MSE: {mse:.6f}")

# ── 시각화 ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle(f'FAST 토크나이저: 인코딩 → 디코딩 (K={tokenizer.K}, n_bins={tokenizer.n_bins})')

for d in range(D):
    ax = axes[d // 4][d % 4]
    ax.plot(t, test_actions[:, d], 'b-', linewidth=1.5, label='원본')
    ax.plot(t, actions_rec[:, d], 'r--', linewidth=1.5, label='복원')
    mse_d = np.mean((test_actions[:, d] - actions_rec[:, d]) ** 2)
    ax.set_title(f'{DIM_NAMES[d]}\nMSE={mse_d:.5f}')
    ax.legend(fontsize=7)

ax_last = axes[1][3]
ax_last.hist(token_ids, bins=50, color='steelblue', edgecolor='white')
ax_last.set_xlabel('토큰 ID')
ax_last.set_ylabel('빈도')
ax_last.set_title('토큰 ID 분포')

plt.tight_layout()
plt.savefig('fast_step3_tokenization.png', dpi=100)
plt.show()
print("→ fast_step3_tokenization.png 저장됨")
