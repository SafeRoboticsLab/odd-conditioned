"""Sloshy / rigid payload generator — MJCF (mjlab / MuJoCo 3.x) port of LINC-POC's PyBullet design.

LINC-POC (`linc/envs/pybullet/env_hexapod.py::_gen_urdf`, envtype='spring') built a payload as a VERTICAL
STACK of boxes: block_0 FIXED to the trunk, then each block on a REVOLUTE hinge (alternating roll/pitch
axes so it can arch in any direction), limit ±0.5 rad, viscous `damping=0.05`, injected into a
`<!-- PAYLOAD_PLACEHOLDER -->` in the base URDF before each run. One scalar `payload∈[0,1]` drove the ODD:
mass (→10 kg), height (→0.45 m), and # active blocks (→5). Rigidity was a DISCRETE switch (spring stack
vs a single fixed box, envtype='payload').

MJCF port + the key upgrade for our ODD: rigidity becomes a CONTINUOUS knob via the MuJoCo hinge
`stiffness` (a spring to springref=0). stiffness=0 ⇒ free-swinging water-like slosh (damping only);
stiffness→large ⇒ the stack self-rights and moves as one rigid box. So the ODD axes are:
  * n_layers   — # hinged blocks (LINC's "# active blocks" / height)   [feasibility/CoM]
  * block_mass — mass per block                                        [feasibility]
  * stiffness  — joint spring = RIGIDITY (0=sloshy … large=rigid)       [the novel coupling knob]
  * damping    — joint viscous damping (LINC used 0.05)

`payload_body(...)` returns an ATTACHABLE <body> snippet (drop it under the Go2 trunk body, LINC's
inject-a-snippet pattern). `sample_scene(...)` wraps N stacks on pedestals into a standalone viewable
scene. MuJoCo box `size` is HALF-extents; LINC block 0.15×0.15×0.05 → size="0.075 0.075 0.025".
"""
from typing import List, Optional, Sequence, Tuple


def mass_profile(n_blocks: int, total_mass: float, profile: str = "uniform") -> List[float]:
    """Per-block masses (length n_blocks = 1 fixed base + hinged layers) summing to total_mass.
    The MASS-DISTRIBUTION ODD axis (shifts payload CoM — top-heavy is far harder to stabilize):
    'uniform' | 'top_heavy' (linearly ↑ up the stack) | 'bottom_heavy' (linearly ↓)."""
    if profile == "uniform":
        w = [1.0] * n_blocks
    elif profile == "top_heavy":
        w = [float(i + 1) for i in range(n_blocks)]
    elif profile == "bottom_heavy":
        w = [float(n_blocks - i) for i in range(n_blocks)]
    else:
        raise ValueError(profile)
    s = sum(w)
    return [total_mass * wi / s for wi in w]

HW = 0.075          # box half-width/depth (LINC 0.15 full)
HH = 0.025          # box half-height     (LINC 0.05 full)
LAYER_DZ = 2 * HH   # vertical spacing between block frames (block height)


def payload_body(name: str = "payload", pos: Tuple[float, float, float] = (0, 0, 0),
                 n_layers: int = 4, block_mass: float = 2.0,
                 masses: Optional[Sequence[float]] = None,
                 stiffness: float = 0.0, damping: float = 0.05, rng: float = 0.5) -> str:
    """MJCF <body> for a sloshy stack: block_0 rigidly on the mount, then `n_layers` hinged blocks.
    Attach under any parent body (the Go2 trunk) at `pos`. ODD axes: n_layers (height/CoM);
    total mass + DISTRIBUTION via `masses` (length n_layers+1, e.g. from mass_profile) else uniform
    block_mass; stiffness (RIGIDITY: 0=sloshy…large=rigid); damping."""
    m = list(masses) if masses is not None else [block_mass] * (n_layers + 1)
    assert len(m) == n_layers + 1, f"masses must have length n_layers+1={n_layers + 1}, got {len(m)}"
    colors = [".85 .75 .1 1", ".2 .7 .35 1"]                      # LINC alternating Yellow/Green
    lines: List[str] = [f'<body name="{name}_mount" pos="{pos[0]} {pos[1]} {pos[2]}">']
    # block_0 — FIXED to the mount (a geom directly on this body == LINC's fixed body_payload joint)
    lines.append(f'  <geom name="{name}_b0" type="box" size="{HW} {HW} {HH}" contype="0" conaffinity="0" '
                 f'pos="0 0 {HH}" mass="{m[0]}" rgba="{colors[0]}"/>')
    indent = "  "
    for i in range(1, n_layers + 1):
        axis = "1 0 0" if i % 2 else "0 1 0"                      # alternate roll/pitch (LINC)
        lines.append(f'{indent}<body name="{name}_b{i}" pos="0 0 {LAYER_DZ}">')
        lines.append(f'{indent}  <joint name="{name}_j{i}" type="hinge" axis="{axis}" '
                     f'range="{-rng} {rng}" limited="true" stiffness="{stiffness}" damping="{damping}"/>')
        lines.append(f'{indent}  <geom name="{name}_b{i}" type="box" size="{HW} {HW} {HH}" contype="0" '
                     f'conaffinity="0" pos="0 0 {HH}" mass="{m[i]}" rgba="{colors[i % 2]}"/>')
        indent += "  "
    for i in range(n_layers):                                    # close nested <body> tags
        indent = indent[:-2]
        lines.append(f'{indent}</body>')
    lines.append('</body>')
    return "\n".join(lines)


def sample_scene(configs: List[dict], spacing: float = 0.6, pedestal_h: float = 0.25) -> str:
    """Standalone MJCF: one payload stack per config, on a pedestal, side by side. `configs` = list of
    dicts passed to payload_body (plus optional 'label')."""
    bodies = []
    for k, cfg in enumerate(configs):
        x = (k - (len(configs) - 1) / 2) * spacing
        name = cfg.get("name", f"pl{k}")
        c = {kk: vv for kk, vv in cfg.items() if kk not in ("label",)}
        c["name"] = name
        # a static pedestal (part of worldbody, no joint) with the stack welded on its top
        stack = payload_body(**{**c, "pos": (0, 0, pedestal_h)})
        # indent the stack under the pedestal body
        stack_indented = "\n".join("    " + ln for ln in stack.splitlines())
        bodies.append(f'''  <body name="{name}_pedestal" pos="{x:.3f} 0 0">
    <geom type="cylinder" size="0.05 {pedestal_h/2}" pos="0 0 {pedestal_h/2}" rgba=".3 .3 .35 1"/>
{stack_indented}
  </body>''')
    return f'''<mujoco model="sloshy_payload_sample">
  <compiler angle="radian"/>   <!-- LINC hinge ranges are RADIANS; MuJoCo defaults to degrees -->
  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="720"/><headlight diffuse=".6 .6 .6"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <geom name="floor" type="plane" size="3 3 .1" rgba=".9 .9 .92 1"/>
    <camera name="side" pos="0 -2.2 0.9" xyaxes="1 0 0 0 0.4 1"/>
{chr(10).join(bodies)}
  </worldbody>
</mujoco>'''


if __name__ == "__main__":
    import os
    # the rigidity-ODD spectrum: free/sloshy -> lightly-returning -> rigid  (n_layers=4, 2 kg/block)
    configs = [
        dict(name="sloshy", stiffness=0.0,   label="stiffness 0 (water-like)"),
        dict(name="medium", stiffness=20.0,  label="stiffness 20 (jelly)"),
        dict(name="rigid",  stiffness=400.0, label="stiffness 400 (~rigid box)"),
    ]
    xml = sample_scene(configs)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "results/payload_sample/sloshy_vs_rigid.xml")
    with open(out, "w") as f:
        f.write(xml)
    print("wrote", out)
    print("view:  MUJOCO_GL=glfw python -m mujoco.viewer --mjcf", out)
