#!/usr/bin/env python3
"""Single-file PPO for continuous control, CleanRL style (adapted to Gymnasium 1.x).

Everything (rollout, GAE, clipped objective) is in this one file so you can read
and modify the algorithm directly. Based on CleanRL's ppo_continuous_action.py
(MIT, https://github.com/vwxyzjn/cleanrl).

    python ppo_continuous.py --env HalfCheetah-v5 --total-timesteps 1000000
    python ppo_continuous.py --env Hopper-v5 --num-envs 4 --capture-video

Logs go to $RL_RUNS_DIR/<run>/ (TensorBoard: charts/episodic_return, losses/*).
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="HalfCheetah-v5")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=2048, help="rollout length per env")
    p.add_argument("--anneal-lr", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu", help="cpu is usually fastest for these small MLPs")
    p.add_argument("--capture-video", action="store_true", help="record videos of env 0 to <run>/videos")
    p.add_argument("--runs-dir", default=os.environ.get("RL_RUNS_DIR", "./runs"))
    p.add_argument("--run-name", default=None)
    a = p.parse_args()
    a.batch_size = a.num_envs * a.num_steps
    a.minibatch_size = a.batch_size // a.num_minibatches
    a.num_iterations = max(a.total_timesteps // a.batch_size, 1)
    return a


def make_env(env_id, idx, capture_video, video_dir, gamma):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video and idx == 0 else None)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, str(video_dir), episode_trigger=lambda e: e % 50 == 0)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10), env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10, 10)))
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
                                    layer_init(nn.Linear(64, 64)), nn.Tanh(),
                                    layer_init(nn.Linear(64, 1), std=1.0))
        self.actor_mean = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
                                        layer_init(nn.Linear(64, 64)), nn.Tanh(),
                                        layer_init(nn.Linear(64, act_dim), std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        dist = Normal(mean, self.actor_logstd.expand_as(mean).exp())
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x)


def main():
    a = parse_args()
    name = a.run_name or f"{a.env}__ppo_cleanrl__s{a.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
    run = Path(a.runs_dir) / name
    if run.exists():
        run = run.with_name(f"{name}__{time.strftime('%H%M%S')}")
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({**vars(a), "algo": "ppo_cleanrl"}, indent=2))
    writer = SummaryWriter(str(run / "tb"))
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(a).items()))

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"run dir: {run}\ndevice: {device}")

    # SAME_STEP autoreset: the step that ends an episode already returns the reset obs,
    # matching the classic CleanRL/Gym 0.26 semantics this algorithm was written for.
    envs = gym.vector.SyncVectorEnv(
        [make_env(a.env, i, a.capture_video, run / "videos", a.gamma) for i in range(a.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "continuous action spaces only"
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=a.learning_rate, eps=1e-5)

    obs = torch.zeros((a.num_steps, a.num_envs, obs_dim), device=device)
    actions = torch.zeros((a.num_steps, a.num_envs, act_dim), device=device)
    logprobs = torch.zeros((a.num_steps, a.num_envs), device=device)
    rewards = torch.zeros((a.num_steps, a.num_envs), device=device)
    dones = torch.zeros((a.num_steps, a.num_envs), device=device)
    values = torch.zeros((a.num_steps, a.num_envs), device=device)

    global_step, start = 0, time.time()
    next_obs, _ = envs.reset(seed=a.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(a.num_envs, device=device)
    recent_returns = []

    for iteration in range(1, a.num_iterations + 1):
        if a.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / a.num_iterations) * a.learning_rate

        for step in range(a.num_steps):
            global_step += a.num_envs
            obs[step], dones[step] = next_obs, next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step], actions[step], logprobs[step] = value.flatten(), action, logprob

            next_obs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(np.logical_or(term, trunc), dtype=torch.float32, device=device)

            # With SAME_STEP autoreset, finished-episode stats live in infos["final_info"].
            ep_info = infos.get("final_info", infos)
            if "episode" in ep_info:
                for i in np.flatnonzero(ep_info["_episode"]):
                    ret, length = float(ep_info["episode"]["r"][i]), int(ep_info["episode"]["l"][i])
                    recent_returns = (recent_returns + [ret])[-20:]
                    writer.add_scalar("charts/episodic_return", ret, global_step)
                    writer.add_scalar("charts/episodic_length", length, global_step)

        # Generalized advantage estimation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(a.num_steps)):
                nextnonterminal = 1.0 - (next_done if t == a.num_steps - 1 else dones[t + 1])
                nextvalues = next_value if t == a.num_steps - 1 else values[t + 1]
                delta = rewards[t] + a.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + a.gamma * a.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs, b_logprobs = obs.reshape(-1, obs_dim), logprobs.reshape(-1)
        b_actions, b_adv = actions.reshape(-1, act_dim), advantages.reshape(-1)
        b_returns, b_values = returns.reshape(-1), values.reshape(-1)

        b_inds, clipfracs = np.arange(a.batch_size), []
        for _ in range(a.update_epochs):
            np.random.shuffle(b_inds)
            for s in range(0, a.batch_size, a.minibatch_size):
                mb = b_inds[s:s + a.minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > a.clip_coef).float().mean().item())
                mb_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio,
                                    -mb_adv * torch.clamp(ratio, 1 - a.clip_coef, 1 + a.clip_coef)).mean()
                newvalue = newvalue.view(-1)
                v_clipped = b_values[mb] + torch.clamp(newvalue - b_values[mb], -a.clip_coef, a.clip_coef)
                v_loss = 0.5 * torch.max((newvalue - b_returns[mb]) ** 2, (v_clipped - b_returns[mb]) ** 2).mean()
                loss = pg_loss - a.ent_coef * entropy.mean() + v_loss * a.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), a.max_grad_norm)
                optimizer.step()
            if a.target_kl is not None and approx_kl > a.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        sps = int(global_step / (time.time() - start))
        for k, v in {"charts/learning_rate": optimizer.param_groups[0]["lr"], "losses/value_loss": v_loss.item(),
                     "losses/policy_loss": pg_loss.item(), "losses/entropy": entropy.mean().item(),
                     "losses/approx_kl": approx_kl.item(), "losses/clipfrac": np.mean(clipfracs),
                     "losses/explained_variance": np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y,
                     "charts/SPS": sps}.items():
            writer.add_scalar(k, v, global_step)
        mean_ret = f"{np.mean(recent_returns):.1f}" if recent_returns else "n/a"
        print(f"iter {iteration}/{a.num_iterations}  step {global_step}  SPS {sps}  return(last20) {mean_ret}")

    torch.save({"agent": agent.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim, "args": vars(a)},
               run / "agent.pt")
    print(f"saved {run / 'agent.pt'}")
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
