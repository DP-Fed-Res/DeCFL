# opacus_accountant.py
import argparse
from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.prv import PRVAccountant

def compute_epsilon_opacus(q: float, sigma: float, steps: int, delta: float) -> float:
    """
    使用 Opacus 的 RDP 会计器计算 (ε, δ)-DP 的 ε
    假设：每步独立 Poisson 子采样率 q，噪声乘数 sigma，高斯机制作用于裁剪后的平均梯度
    """
    # acct = RDPAccountant()
    acct = PRVAccountant()
    # 将同一机制重复 steps 次加入会计器
    for _ in range(steps):
        acct.step(noise_multiplier=sigma, sample_rate=q)
    eps = acct.get_epsilon(delta=delta)
    return eps


if __name__ == "__main__":
    q = 0.02
    sigma = 1.0
    clip = 1.0
    local_steps = 380
    rounds = 1
    delta = 1e-5
    eps = compute_epsilon_opacus(q, sigma * clip, local_steps * rounds, delta)
    print(f"[Opacus] q={q}, noise_sigma={sigma * clip}, steps={local_steps * rounds}, delta={delta} -> epsilon={eps:.4f}")