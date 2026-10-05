"""Where trained policies are read from and results are written to.

    ODD_CHECKPOINTS   trained policies, one directory per policy      (default: <repo>/checkpoints)
    ODD_OUTPUTS       results, figures and videos                     (default: <repo>/outputs)

A policy directory holds ``model.zip`` (the reach-avoid PPO twin), ``tensornormalize.pt`` (its frozen
observation statistics, found automatically next to the model) and ``config.yaml`` (the training config).
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.abspath(os.path.expanduser(os.environ.get("ODD_CHECKPOINTS", os.path.join(REPO, "checkpoints"))))
OUTPUTS = os.path.abspath(os.path.expanduser(os.environ.get("ODD_OUTPUTS", os.path.join(REPO, "outputs"))))


def checkpoint(name: str) -> str:
    """Path of a trained policy: a name from ``policies.POLICIES``, or an explicit ``.zip`` path."""
    path = name if name.endswith(".zip") else os.path.join(CHECKPOINTS, name, "model.zip")
    if not os.path.exists(path):
        hint = (f"fetch the published weights (bash scripts/fetch_weights.sh) or train it "
                f"(bash scripts/train.sh {name})") if not name.endswith(".zip") else "no such file"
        raise SystemExit(f"missing checkpoint {path}\n  {hint}")
    return path


def output_dir(*parts: str) -> str:
    """``$ODD_OUTPUTS/<parts...>``, created if needed."""
    d = os.path.join(OUTPUTS, *parts)
    os.makedirs(d, exist_ok=True)
    return d
