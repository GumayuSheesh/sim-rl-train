#!/usr/bin/env python3
"""Train a Stable-Baselines3 / sb3-contrib agent on any Gymnasium environment.

Each run gets its own directory under $RL_RUNS_DIR (default /workspace/runs):
    <run>/config.json          arguments + library versions
    <run>/tb/                  TensorBoard logs (TensorBoard on port 6006 reads $RL_RUNS_DIR)
    <run>/checkpoints/         periodic checkpoints (+ VecNormalize stats)
    <run>/best/best_model.zip  best model by periodic evaluation
    <run>/final_model.zip      model at the end (also written on Ctrl-C)

Examples:
    python train_sb3.py --env HalfCheetah-v5 --algo sac --timesteps 1000000
    python train_sb3.py --env Hopper-v5 --algo ppo --n-envs 8 --normalize
    python train_sb3.py --resume /workspace/runs/<run>/checkpoints/<ckpt>.zip --timesteps 500000

Device tip: PPO/A2C with small MLP policies are usually faster on CPU (--device cpu,
many --n-envs); SAC/TD3/TQC with large batches benefit from the GPU.
"""
import argparse
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import stable_baselines3 as sb3
import torch
from sb3_contrib import TQC
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

ALGOS = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3, "tqc": TQC}
ON_POLICY = {"ppo", "a2c"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="HalfCheetah-v5", help="Gymnasium env id")
    p.add_argument("--algo", default="ppo", choices=sorted(ALGOS))
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=None, help="parallel envs (default: 8 on-policy, 1 off-policy)")
    p.add_argument("--subproc", action="store_true", help="run envs in subprocesses (helps with many envs/CPU cores)")
    p.add_argument("--normalize", action="store_true", help="VecNormalize observations and rewards")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=None, help="learning rate override")
    p.add_argument("--hparams", default="{}", help='JSON of extra algo kwargs, e.g. \'{"batch_size":512}\'')
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", default=os.environ.get("RL_RUNS_DIR", "./runs"))
    p.add_argument("--checkpoint-freq", type=int, default=100_000, help="in total env steps")
    p.add_argument("--eval-freq", type=int, default=25_000, help="in total env steps; 0 disables")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--resume", default=None, help="path to a saved .zip model to continue training")
    return p.parse_args()


def main():
    a = parse_args()
    n_envs = a.n_envs or (8 if a.algo in ON_POLICY else 1)
    name = a.run_name or f"{a.env}__{a.algo}__s{a.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
    run = Path(a.runs_dir) / name
    if run.exists():  # never overwrite a previous run's results
        run = run.with_name(f"{name}__{time.strftime('%H%M%S')}")
    (run / "checkpoints").mkdir(parents=True)

    vec_cls = SubprocVecEnv if a.subproc and n_envs > 1 else None
    env = make_vec_env(a.env, n_envs=n_envs, seed=a.seed, vec_env_cls=vec_cls)
    eval_env = make_vec_env(a.env, n_envs=1, seed=a.seed + 10_000)
    if a.normalize:
        env = VecNormalize(env)
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
        if a.resume:
            r = Path(a.resume)  # model_N_steps.zip -> model_vecnormalize_N_steps.pkl
            stats = (r.with_name(r.stem.replace("model_", "model_vecnormalize_", 1) + ".pkl")
                     if r.stem.startswith("model_") else r.parent / "final_vecnormalize.pkl")
            if stats.exists():
                env = VecNormalize.load(str(stats), env.venv)
                print(f"loaded normalization stats {stats}")

    algo = ALGOS[a.algo]
    kwargs = json.loads(a.hparams)
    if a.lr is not None:
        kwargs["learning_rate"] = a.lr
    if a.resume:
        model = algo.load(a.resume, env=env, device=a.device, tensorboard_log=str(run / "tb"), **kwargs)
        print(f"resumed from {a.resume} at {model.num_timesteps} steps")
    else:
        model = algo("MlpPolicy", env, device=a.device, seed=a.seed, verbose=1,
                     tensorboard_log=str(run / "tb"), **kwargs)

    config = {**vars(a), "n_envs": n_envs, "run_dir": str(run), "device_used": str(model.device),
              "versions": {"sb3": sb3.__version__, "gymnasium": gym.__version__, "torch": torch.__version__,
                           "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    (run / "config.json").write_text(json.dumps(config, indent=2))
    print(f"run dir: {run}\ndevice: {model.device}  envs: {n_envs}")

    callbacks = [CheckpointCallback(save_freq=max(a.checkpoint_freq // n_envs, 1), save_path=str(run / "checkpoints"),
                                    name_prefix="model", save_vecnormalize=a.normalize)]
    if a.eval_freq > 0:
        # EvalCallback syncs VecNormalize stats from env to eval_env before each evaluation.
        callbacks.append(EvalCallback(eval_env, best_model_save_path=str(run / "best"), log_path=str(run / "eval"),
                                      eval_freq=max(a.eval_freq // n_envs, 1), n_eval_episodes=a.eval_episodes,
                                      deterministic=True))
    try:
        model.learn(total_timesteps=a.timesteps, callback=callbacks, tb_log_name="tb",
                    reset_num_timesteps=not a.resume, progress_bar=True)
    except KeyboardInterrupt:
        print("interrupted: saving final model")
    finally:
        model.save(run / "final_model")
        if a.normalize:
            env.save(str(run / "final_vecnormalize.pkl"))
        env.close()
        eval_env.close()
        print(f"saved {run / 'final_model.zip'} ({model.num_timesteps} steps)")


if __name__ == "__main__":
    main()
