"""E007 — Why is Q(upright, still) = -0.62 when the true avoid value is 1.0?

E006 died on this. Training was HEALTHY (ep_len 13.5->161, ep_rew 6.03->139.5, critic_loss
0.077->2.3e-6) but every learned safe set scored EMPTY, because V < 0 everywhere. The critic
learned the right SHAPE (-0.62 upright > -0.99 at theta=0.9) at the wrong LEVEL.

The suspect (Buzi concurs): my `l_neg` argument may be INVERTED.

  The claim was: a constant l below every reachable V' makes max(l, gamma V') = gamma V', so
  the reach-avoid backup min(g, max(l, gamma V')) collapses to the avoid backup min(g, gamma V').

  The counter-claim: reach-avoid with a constant negative l is ALSO literally "a target you can
  never reach" -- and the RA value of a never-reachable target is l, everywhere. The probe is
  suggestive: Q lands on -0.99 ~= l_neg=-1 at exactly the doomed states.

THE DECISIVE TEST
-----------------
`SafetySAC` is single-player and IGNORES l entirely (backup: min(g, V')). Run the two side by
side on the same env, same seed, and watch V(upright, still) -- a state whose true avoid value
is trivially g = 1.0.

  * SafetySAC recovers ~1.0 and IsaacsSAC does not  => l_neg is the culprit; my collapse
    argument is wrong and the avoid-vs-reach-avoid formulation must change.
  * BOTH sit at ~-0.6                               => l is innocent; the bug is in the margin
    wiring or the terminal/truncation handling, and it would have poisoned any arm.
  * BOTH recover ~1.0                               => neither; the E006 scoring path is wrong
    (it queries policy.actor + policy.dstb_actor and concatenates -- that is the only piece
    the diagnostic below does NOT share with E006).

Also sweeps l_neg for IsaacsSAC: if the RA-with-unreachable-target theory is right, V should
TRACK l_neg (l=-5 => V~-5). That is a falsifiable prediction, not a vibe.

NOTE the adversary asymmetry, and why it does not confound the test: SafetySAC runs with
adversary=False (F=0), so it solves an EASIER game than IsaacsSAC (F_bar=2). But upright-and-
still is safe in BOTH games -- control accel u_max/(m l^2) = 10 m/s^2 dwarfs disturbance accel
F_bar/(m l) = 1 m/s^2 at m=2 -- so both should give V(upright) ~ 1.0. The test is about the
LEVEL, not the game.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from odd_conditioned.envs.pendulum_odd import PendulumODD  # noqa: E402

STEPS = int(os.environ.get("E007_STEPS", 25_000))
PROBE = [(0.0, 0.0), (0.1, 0.0), (0.3, 0.0), (0.9, 0.0)]   # upright -> doomed


def make(adversary, l_neg):
    from stable_baselines3.common.monitor import Monitor
    return Monitor(PendulumODD(odd_mode="static", obs_mode="oracle",
                               mass_range=(2.0, 2.0), resample_prob=0.0,   # pin the ODD: this
                               margin_mode="avoid", l_neg=l_neg,           # is about the LEVEL
                               adversary=adversary, seed=0))


def probe(model, two_player, mass=2.0):
    obs = np.array([[np.sin(t), np.cos(t), w, mass] for t, w in PROBE], dtype=np.float32)
    t = torch.as_tensor(obs)
    with torch.no_grad():
        a = model.policy.actor(t, deterministic=True)
        if two_player:
            a = torch.cat([a, model.policy.dstb_actor(t, deterministic=True)], dim=1)
        q = model.critic(t, a)
        q = torch.min(*q) if isinstance(q, (list, tuple)) else q
    return q.squeeze(-1).numpy()


def run(name, cls, two_player, l_neg, **kw):
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: make(adversary=two_player, l_neg=l_neg)])
    m = cls("MlpPolicy", env, learning_rate=5e-4,
            policy_kwargs=dict(net_arch=[128, 128, 128]),
            gamma=0.99, verbose=0, seed=0, device="cpu", **kw)
    m.learn(total_timesteps=STEPS, progress_bar=False)
    v = probe(m, two_player)
    print(f"  {name:34s} " + "  ".join(f"{x:+7.3f}" for x in v))
    return v


def main():
    from safety_sb3 import SafetySAC, IsaacsSAC

    print(f"V(theta, omega) after {STEPS:,} steps, mass pinned at 2.0 kg.")
    print("TRUE avoid value at upright-and-still = g = 1.0  (trivially safe in both games).\n")
    print(f"  {'':34s} " + "  ".join(f"({t:+.1f},{w:+.1f})" for t, w in PROBE))
    print("  " + "-" * 74)

    v_safety = run("SafetySAC (1p, IGNORES l)", SafetySAC, False, -1.0)
    v_isaacs = {}
    for l in (-1.0, -5.0):
        v_isaacs[l] = run(f"IsaacsSAC  (2p, l_neg={l:g})", IsaacsSAC, True, l, ctrl_action_dim=1)

    up_s, up_i1, up_i5 = v_safety[0], v_isaacs[-1.0][0], v_isaacs[-5.0][0]
    print("\n" + "=" * 78)
    tracks_l = abs(up_i5 - up_i1) > 0.5      # did V follow l_neg from -1 to -5?
    if up_s > 0.5 and up_i1 < 0.0:
        print("VERDICT: l_neg IS THE CULPRIT. SafetySAC (which ignores l) recovers V(upright)~1;")
        print("         IsaacsSAC with a constant l does not. The 'l_neg collapses reach-avoid to")
        print("         avoid' argument is WRONG -- a constant l is a target you can never reach,")
        print("         and RA of an unreachable target is l. E006's avoid formulation must change.")
        if tracks_l:
            print(f"         CONFIRMED by the sweep: V(upright) tracks l_neg ({up_i1:+.3f} -> {up_i5:+.3f}).")
    elif up_s < 0.0 and up_i1 < 0.0:
        print("VERDICT: l IS INNOCENT -- BOTH sit low. The bug is upstream of the two-player")
        print("         machinery: the margin wiring or the terminal/truncation handling. It")
        print("         would have poisoned every arm. Check that nt=0 ONLY on true failure")
        print("         (g<0) and never on timeout, and that reward really is g.")
    elif up_s > 0.5 and up_i1 > 0.5:
        print("VERDICT: NEITHER -- both learn the right level. The bug is in E006's SCORING path,")
        print("         which is the only piece this diagnostic does not share (it builds the")
        print("         action as concat(actor, dstb_actor) before calling critic).")
    else:
        print("VERDICT: inconclusive -- read the table above.")
    print("=" * 78)


if __name__ == "__main__":
    main()
