"""Render the two specialists under the 0.35 lateral pull, side by side — dodge vs brace."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch as th, imageio.v2 as imageio
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize
RUNS="results/go2_payload_runs"; DEV="cuda:0"; PULL=th.tensor([0.,1.,0.])
def render_run(task, steps=200, nenv=2):
    d=f"{RUNS}/{task}_gameplaysac"
    env=make_tensor(task, nenv, DEV, adversary=True, render_mode="rgb_array")
    try: env.mj.cfg.viewer.max_extra_envs=max(1,nenv-1)
    except Exception: pass
    Algo=getattr(safety_sb3, algo_name(task,adversary=True).replace("PPO","SAC"))
    model=Algo.load(f"{d}/final_model.zip", env=env, device=DEV,
                    custom_objects={"_use_lb":False,"_lb_dir":"/tmp/lb","tensorboard_log":None,"buffer_size":1})
    norm=TensorVecNormalize.load(f"{d}/tensornormalize.pt", env); norm.training=False
    dd=spec(task).dstb_dim
    env.force_scale=0.35*th.ones(nenv,device=DEV)
    dstb=(PULL/PULL.norm()).to(DEV)[None].expand(nenv,dd).contiguous()
    obs=env.reset(); frames=[]
    for k in range(steps):
        with th.no_grad(): a=th.clamp(model.policy._predict(norm.normalize_obs(obs),deterministic=True),-1,1)
        obs,*_=env.step_tensor(th.cat([a,dstb],dim=1))
        frames.append(np.asarray(env.render()))
    env.close(); return frames
def label(frames, txt):
    out=[]
    for f in frames:
        f=f.copy()
        f[:34,:,:]=(f[:34,:,:]*0.35).astype(f.dtype)  # dim a top band for text
        out.append(f)
    return out
f_light=label(render_run("go2_payload_light_rigid"), "LIGHT+RIGID")
f_heavy=label(render_run("go2_payload_heavy_sloshy"), "HEAVY+SLOSHY")
n=min(len(f_light),len(f_heavy)); h=min(f_light[0].shape[0],f_heavy[0].shape[0]); w=min(f_light[0].shape[1],f_heavy[0].shape[1])
comb=[np.hstack([f_light[i][:h,:w], f_heavy[i][:h,:w]]) for i in range(n)]
imageio.mimsave("results/go2_payload_runs/bifurcation_dodge_vs_brace.mp4", comb, fps=30, macro_block_size=1)
print("wrote bifurcation_dodge_vs_brace.mp4  (%d frames, %dx%d)"%(n, comb[0].shape[1], comb[0].shape[0]))
imageio.imwrite("results/go2_payload_runs/bifurcation_frame.png", comb[n//2])
print("wrote mid frame")
