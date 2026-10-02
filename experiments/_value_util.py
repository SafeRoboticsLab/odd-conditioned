"""E071 value readout — evaluate the STAND twin's reach-avoid value V_stand(x, W) at runtime.

The handoff trigger needs V_stand at the CURRENT sim state = the STAND twin's own value estimate.
The STAND twin is a ReachAvoidPPO2P. Key facts established while wiring this (E071, Task 1):

  * The 2P PPO learner has a STATE-ONLY value net V(s) per player (unlike SAC's Q(s, [a_ctrl, a_dstb]));
    ``self.policy`` is the CONTROL player. Its value net was built over the learner's observation_space,
    which for this training run is the **48-dim ACTOR obs** (Box(48,), NOT the 75-dim mjlab critic group).
    So V_stand is read off the actor-obs surface — the exact same obs the actor consumes — via
    ``policy.predict_values(obs)``. (This is what ``eval.policies.safety_modules`` does for an on-policy
    twin: ``value_fn = policy.predict_values(s_obs).squeeze(-1)``.)
  * NORMALIZATION: ``predict_values`` does NOT standardize its input (SB3's ActorCriticPolicy value net is
    a bare MLP; only image obs get scaled). The twin's obs statistics live in ``tensornormalize.pt`` and are
    applied EXTERNALLY by the ``norm`` callable ``load_twin`` returns (48-dim, clip ±10). So we must pass
    ``norm(obs)`` — the same normalized obs fed to the actor — NOT raw obs.
  * The current actor obs is cached by ``base.py`` on every step/reset as ``env.mj._zoo_last_obs`` (48-dim,
    unnormalized), so V_stand can be read at the current state with no extra env step.

Semantics: reach-avoid ``V >= 0`` == the STAND spec is (estimated) still certifiable from here; ``V < 0`` ==
standing is being given up. Validated in E071 Task 1: under the stand policy at pull 0.2, mean V contracts
monotonically 0.044 (W=0) -> 0.029 (W=150) -> 0.007 (W=200) -> -0.012 (W=250) = the empirical certificate
contraction, crossing 0 in the W~200-250 band.
"""

from __future__ import annotations

import torch


def stand_value(env, model, norm) -> torch.Tensor:
  """V_stand at the CURRENT sim state, shape [N].

  ``env``   the MjlabTensorSafetyEnv the rollout is stepping (any weight-ladder cfg — the actor obs
            surface is identical across stand/rest tasks). Reads ``env.mj._zoo_last_obs`` (the 48-dim
            actor obs cached on the last step/reset).
  ``model`` the STAND twin (ReachAvoidPPO2P) from ``load_twin``.
  ``norm``  the twin's obs normalizer from ``load_twin`` (48-dim, clip ±10).
  """
  obs = env.mj._zoo_last_obs
  with torch.no_grad():
    return model.policy.predict_values(norm(obs)).squeeze(-1)
