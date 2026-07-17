"""Periodic scoring of a learned ODD-conditioned critic against exact grid ground truth.

WHY THIS EXISTS
---------------
For a normal RL run you watch return. Here return is meaningless -- the reward IS the safety
margin, so a high return only says "did not fall", not "did the critic learn the right SET".
The thing we actually care about is the zero-superlevel set of V_hat vs. the grid's, per ODD
rung. That is what this logs, and it is the project-log skill's "eval video" analogue: a
periodic artifact you can watch develop, uploaded roughly every 10-20 min of wall-clock.

THREE SIGNALS, KEPT SEPARATE ON PURPOSE
---------------------------------------
  vol_ratio     -- learned/true safe-set volume per rung. COLLAPSE shows up here as the ratio
                   falling toward the worst rung's while the easy rungs are starved.
  optimism      -- learned says SAFE where truth says UNSAFE. THE UNSAFE DIRECTION. Expected
                   from adversary staleness: with the ODD sampled across the family each rung
                   gets a fraction of the adversary's updates and the best-response
                   disturbance differs per rung, so the critic goes optimistic at the HARD
                   rungs. Never average this into an IoU -- it would hide the failure that
                   matters (borrowing PBF's h*-sign device: one signed scalar, two meanings).
  conservatism  -- learned says UNSAFE where truth says SAFE. Costs liveness, not safety.

  value_loss    -- logged by SB3, but watch it here: a fresh critic + hard domain
                   randomization diverges (1e9 -> 1e14, unrecoverable, "looks like a
                   curriculum drop but isn't"). ODD-CONDITIONING IS A FORM OF DR, so this is a
                   PREDICTED failure of the first conditioned run. Healthy ~ 1e-2.

WEIGHTS STAY LOCAL. Never wandb.save() a checkpoint, never upload a weight artifact.
"""

import os
from typing import Optional, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ODDSafeSetEval(BaseCallback):
    def __init__(
        self,
        truth_npz: str,
        rungs: Sequence[float],
        obs_mode: str,
        eval_freq: int = 25_000,        # ~15 min wall-clock at the ~40 steps/s measured on CPU
        use_wandb: bool = True,
        fig_dir: Optional[str] = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        d = np.load(truth_npz)
        self.th, self.om, self.sweep, self.Vtrue = d["theta"], d["omega"], d["sweep"], d["V"]
        self.cell = (self.th[1] - self.th[0]) * (self.om[1] - self.om[0])
        self.rungs, self.obs_mode = list(rungs), obs_mode
        self.eval_freq, self.use_wandb, self.fig_dir = eval_freq, use_wandb, fig_dir
        if fig_dir:
            os.makedirs(fig_dir, exist_ok=True)

    def _truth_at(self, m):
        return self.Vtrue[int(np.argmin(np.abs(self.sweep - m)))]

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        from experiments.E006_conditioned_critic import learned_safe_set, compare

        rows, figs = {}, []
        for m in self.rungs:
            V = learned_safe_set(self.model, self.th, self.om, [m], self.obs_mode,
                                 device=self.model.device)
            s = compare(V, self._truth_at(m), self.cell)
            rows[m] = s
            for k, v in s.items():
                self.logger.record(f"odd/{k}_m{m:g}", v)
            figs.append((m, V, self._truth_at(m)))

        # aggregate: the two numbers that decide the experiment
        self.logger.record("odd/mean_vol_ratio", float(np.mean([r["vol_ratio"] for r in rows.values()])))
        self.logger.record("odd/total_optimism", float(np.sum([r["optimism"] for r in rows.values()])))

        if self.verbose:
            msg = "  ".join(f"m={m:g}: ratio={r['vol_ratio']:.2f} opt={r['optimism']:.2f}"
                            for m, r in rows.items())
            print(f"[{self.num_timesteps:>8,}] {msg}", flush=True)

        if self.use_wandb:
            self._log_overlay(figs)
        return True

    def _log_overlay(self, figs):
        """The eval-artifact analogue: learned vs true safe set per rung, uploaded to wandb."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import wandb
        except Exception:
            return

        fig, axes = plt.subplots(1, len(figs), figsize=(3.4 * len(figs), 3.6), squeeze=False)
        for ax, (m, Vl, Vt) in zip(axes[0], figs):
            ax.contourf(self.th, self.om, (Vt >= 0).T, levels=[0.5, 1.5], colors=["#2e6b4a"], alpha=0.45)
            ax.contour(self.th, self.om, (Vt >= 0).T, levels=[0.5], colors=["#1d4430"], linewidths=1.8)
            ax.contour(self.th, self.om, (Vl >= 0).T, levels=[0.5], colors=["#c0392b"], linewidths=2.0)
            ax.set_title(f"m={m:g} kg", fontsize=10)
            ax.set_xlabel(r"$\theta$")
        axes[0][0].set_ylabel(r"$\omega$")
        fig.suptitle(f"green: grid ground truth   |   red: learned $\\hat V=0$   "
                     f"(step {self.num_timesteps:,})", fontsize=11)
        fig.tight_layout()
        wandb.log({"odd/safe_set_overlay": wandb.Image(fig)}, step=self.num_timesteps)
        if self.fig_dir:
            fig.savefig(os.path.join(self.fig_dir, f"overlay_{self.num_timesteps:09d}.png"), dpi=110)
        plt.close(fig)
