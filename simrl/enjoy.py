#!/usr/bin/env python3
"""Evaluate a trained SB3 model and record an MP4 of its episodes (headless).

    python enjoy.py /workspace/runs/<run>/best/best_model.zip --episodes 3
    python enjoy.py <model.zip> --env Hopper-v5 --algo sac --no-video

Env id and algorithm are read from <run>/config.json when available.
VecNormalize stats (final_vecnormalize.pkl or checkpoint .pkl) are applied if found.
"""
import argparse
import json
import pickle
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from sb3_contrib import TQC
from stable_baselines3 import A2C, PPO, SAC, TD3

ALGOS = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3, "tqc": TQC}


def find_run_config(model_path: Path):
    for d in [model_path.parent, model_path.parent.parent]:
        if (d / "config.json").exists():
            return d, json.loads((d / "config.json").read_text())
    return model_path.parent, {}


def find_obs_rms(model_path: Path, run_dir: Path):
    # checkpoints: model_N_steps.zip -> model_vecnormalize_N_steps.pkl; otherwise the run's final stats
    cands = [model_path.with_name(model_path.stem.replace("model_", "model_vecnormalize_", 1) + ".pkl"),
             run_dir / "final_vecnormalize.pkl"]
    for c in cands:
        if c.exists():
            with open(c, "rb") as f:
                vn = pickle.load(f)
            return vn.obs_rms, vn.clip_obs, vn.epsilon, c
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--env", default=None)
    p.add_argument("--algo", default=None, choices=sorted(ALGOS))
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--fps", type=int, default=30)
    a = p.parse_args()

    model_path = Path(a.model)
    run_dir, cfg = find_run_config(model_path)
    env_id = a.env or cfg.get("env")
    algo = a.algo or cfg.get("algo")
    if not env_id or not algo:
        p.error("could not find config.json next to the model; pass --env and --algo")

    model = ALGOS[algo].load(str(model_path), device="cpu")
    norm = find_obs_rms(model_path, run_dir)
    if norm:
        print(f"using observation normalization from {norm[3]}")

    def prep(obs):
        if not norm:
            return obs
        rms, clip, eps, _ = norm
        return np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip)

    env = gym.make(env_id, render_mode=None if a.no_video else "rgb_array")
    frames, returns = [], []
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=a.seed + ep)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(prep(obs), deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += float(r)
            done = term or trunc
            if not a.no_video:
                frames.append(env.render())
        returns.append(total)
        print(f"episode {ep}: return {total:.1f}")
    env.close()
    print(f"mean return over {a.episodes} episodes: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")

    if frames:
        out = run_dir / "videos"
        out.mkdir(exist_ok=True)
        path = out / f"{model_path.stem}.mp4"
        i = 1
        while path.exists():  # do not overwrite earlier videos
            path = out / f"{model_path.stem}-{i}.mp4"
            i += 1
        imageio.mimsave(path, frames, fps=a.fps, macro_block_size=1)
        print(f"video: {path}")


if __name__ == "__main__":
    main()
