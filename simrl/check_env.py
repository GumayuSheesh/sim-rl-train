#!/usr/bin/env python3
"""Smoke-test the simulation/RL stack on this machine.

Checks: torch + CUDA device, MuJoCo physics stepping, headless rendering
(MUJOCO_GL), and optionally a short real PPO training run with checkpoint save.
Writes a JSON report and exits non-zero if any check fails.

    python /opt/simrl/check_env.py                   # GPU + render + 2k-step PPO
    python /opt/simrl/check_env.py --require-gpu     # fail if CUDA is unavailable
    python /opt/simrl/check_env.py --train-steps 0   # skip training
"""
import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path


def check(name, fn, report):
    t0 = time.time()
    try:
        info = fn()
        report["checks"][name] = {"ok": True, "seconds": round(time.time() - t0, 2), **(info or {})}
        print(f"[ OK ] {name}: {info}")
    except Exception as exc:  # report every failure, keep going
        report["checks"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                  "traceback": traceback.format_exc(limit=3)}
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="HalfCheetah-v5")
    p.add_argument("--train-steps", type=int, default=2048)
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--out-dir", default=os.path.join(os.environ.get("RL_RUNS_DIR", "./runs"), "_checks"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = {"time": stamp, "python": sys.version.split()[0], "platform": platform.platform(),
              "MUJOCO_GL": os.environ.get("MUJOCO_GL"), "env_id": args.env, "checks": {}}

    def torch_check():
        import torch
        info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                "torch_path": os.path.dirname(torch.__file__)}
        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            a = torch.randn(1024, 1024, device="cuda")
            info["matmul_ok"] = bool(torch.isfinite(a @ a).all().item())
        elif args.require_gpu:
            raise RuntimeError("CUDA not available but --require-gpu was given")
        return info

    def physics_check():
        import gymnasium as gym
        import mujoco
        env = gym.make(args.env)
        obs, _ = env.reset(seed=0)
        t0, n = time.time(), 2000
        for _ in range(n):
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                env.reset()
        sps = n / (time.time() - t0)
        env.close()
        return {"mujoco": mujoco.__version__, "gymnasium": gym.__version__,
                "obs_dim": int(obs.shape[0]), "single_env_steps_per_s": round(sps)}

    def render_check():
        import gymnasium as gym
        import numpy as np
        env = gym.make(args.env, render_mode="rgb_array", width=320, height=240)
        env.reset(seed=0)
        frame = env.render()
        env.close()
        if frame is None or frame.shape != (240, 320, 3) or float(np.std(frame)) < 1.0:
            raise RuntimeError(f"bad frame: {None if frame is None else frame.shape}")
        import imageio.v2 as imageio
        png = out_dir / f"render-{stamp}.png"
        imageio.imwrite(png, frame)
        return {"backend": os.environ.get("MUJOCO_GL"), "frame": list(frame.shape), "png": str(png)}

    def train_check():
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        venv = make_vec_env(args.env, n_envs=2, seed=0)
        model = PPO("MlpPolicy", venv, n_steps=512, batch_size=256, device="cpu", seed=0, verbose=0)
        model.learn(total_timesteps=args.train_steps)
        path = out_dir / f"ppo-smoke-{stamp}.zip"
        model.save(path)
        PPO.load(path, device="cpu")
        venv.close()
        return {"timesteps": model.num_timesteps, "model": str(path)}

    check("torch", torch_check, report)
    check("physics", physics_check, report)
    check("render", render_check, report)
    if args.train_steps > 0:
        check("train", train_check, report)

    report["ok"] = all(c["ok"] for c in report["checks"].values())
    path = out_dir / f"check-{stamp}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {path}\nRESULT: {'PASS' if report['ok'] else 'FAIL'}")
    if not report["checks"].get("render", {}).get("ok", True):
        print("hint: rendering failed; try `export MUJOCO_GL=osmesa` (CPU) and re-run.")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
