# Sim-RL Train — Runpod Pod template

Interactive GPU pod for reinforcement learning in physics simulation:

- MuJoCo 4.0 + Gymnasium 1.4 (HalfCheetah, Hopper, Walker2d, Ant, Humanoid, ... v5 envs)
- Stable-Baselines3 2.9 + sb3-contrib (PPO, A2C, SAC, TD3, TQC)
- A single-file CleanRL-style PPO you can read and modify
- JupyterLab (8888), SSH (22), TensorBoard (6006, auto-started)
- Headless rendering (EGL on the GPU) for frames and MP4 videos

Base image: official `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
(Python 3.12, CUDA 12.8, torch 2.8.0). The base's `/start.sh` is kept as the
container command, so SSH/Jupyter behave exactly like the official PyTorch template.

## Files

    Dockerfile                    image recipe (documented)
    requirements.txt              pinned RL/sim packages (torch is kept from the base)
    post_start.sh                 /post_start.sh hook: starts TensorBoard, seeds examples
    simrl/check_env.py            smoke test: GPU, physics, rendering, short PPO run
    simrl/train_sb3.py            train any SB3 algorithm on any Gymnasium env
    simrl/enjoy.py                evaluate a model and record an MP4
    simrl/ppo_continuous.py       CleanRL-style single-file PPO
    simrl/quickstart.ipynb        notebook walkthrough
    .github/workflows/build-image.yml   optional: build + push to GHCR on GitHub

## 1. Build and push the image

Docker is not needed on your Mac if you use the GitHub workflow:

    cd ~/runpod-sim-train
    git init && git add . && git commit -m "sim-rl template"
    gh repo create sim-rl-train --private --source . --push
    git tag v0.1.0 && git push --tags        # builds ghcr.io/<you>/sim-rl-train:0.1.0

Then make the GHCR package public (GitHub > Packages > sim-rl-train > settings),
or add a registry credential in Runpod (Settings > Container Registry Auth).

With a local x86_64 Docker instead:

    docker build --platform linux/amd64 -t <dockerhub-user>/sim-rl-train:0.1.0 .
    docker push <dockerhub-user>/sim-rl-train:0.1.0

## 2. Register the Runpod template

    runpodctl template create \
      --name "Sim-RL Train (MuJoCo + SB3)" \
      --image ghcr.io/<you>/sim-rl-train:0.1.0 \
      --container-disk-in-gb 40 \
      --volume-in-gb 50 --volume-mount-path /workspace \
      --ports "8888/http,6006/http,22/tcp" \
      --port-labels "8888=JupyterLab,6006=TensorBoard,22=SSH" \
      --env '{"MUJOCO_GL":"egl"}'

(Add `--registry-auth-id <id>` for a private image.) Or in the console:
Templates > New Template, with the same values.

| Setting            | Value                                   | Status |
|--------------------|-----------------------------------------|--------|
| Container image    | your pushed tag (pin the digest later)  | set after build |
| Container command  | leave empty (base `/start.sh`)          | from base source |
| Container disk     | 40 GB                                   | suggested; base template uses 30 GB + ~3 GB of extras |
| Volume             | 50 GB at `/workspace`                   | suggested |
| HTTP ports         | 8888 (Jupyter), 6006 (TensorBoard)      | |
| TCP port           | 22 (SSH)                                | |
| GPU / CUDA         | any NVIDIA GPU, min CUDA 12.8           | |

Environment variables:

| Name                 | Purpose |
|----------------------|---------|
| `JUPYTER_PASSWORD`   | Required for JupyterLab to start (it becomes the login token). The console's "Start Jupyter" option sets it; with `runpodctl` pass it in `--env`. |
| `PUBLIC_KEY`         | Your SSH public key, enables direct SSH on port 22. |
| `MUJOCO_GL`          | `egl` (default, GPU) or `osmesa` (CPU fallback if EGL fails). |
| `DISABLE_TENSORBOARD`| `true` to not start TensorBoard. |
| `RL_RUNS_DIR`        | Where runs/logs go (default `/workspace/runs`). |

Note: the Runpod proxy URL is not authentication. TensorBoard has no login,
so anyone with the pod's 6006 URL can view your training curves.

## 3. Deploy a pod

    runpodctl pod create --template-id <your-template-id> \
      --gpu-id "NVIDIA GeForce RTX 4090" --min-cuda-version 12.8 \
      --env '{"JUPYTER_PASSWORD":"<choose-a-token>","MUJOCO_GL":"egl"}'

Choosing hardware: MuJoCo simulation runs on the CPU. PPO with small MLP policies
is usually faster with `--device cpu` and many environments, so vCPU count matters
as much as the GPU. SAC/TD3/TQC (big replay batches), larger networks and pixel
observations make better use of the GPU. A mid-range GPU (RTX 4090/A5000/L4) is
plenty; pick a pod with many vCPUs for lots of parallel envs.

## 4. Use it

First, check the pod (from the Jupyter terminal or SSH):

    python /opt/simrl/check_env.py --require-gpu

It must print `RESULT: PASS` (torch on CUDA, MuJoCo stepping, a rendered frame,
a 2k-step PPO training run). The report goes to `/workspace/runs/_checks/`.

Train with Stable-Baselines3:

    python /opt/simrl/train_sb3.py --env HalfCheetah-v5 --algo sac --timesteps 1000000
    python /opt/simrl/train_sb3.py --env Hopper-v5 --algo ppo --n-envs 16 --subproc --normalize --device cpu
    python /opt/simrl/train_sb3.py --env Ant-v5 --algo tqc --hparams '{"batch_size":512}'

Each run gets `/workspace/runs/<env>__<algo>__s<seed>__<time>/` with `config.json`,
TensorBoard logs, checkpoints every 100k steps, `best/best_model.zip` and
`final_model.zip` (also written on Ctrl-C). Continue a run:

    python /opt/simrl/train_sb3.py --env Hopper-v5 --algo ppo --normalize \
      --resume /workspace/runs/<run>/checkpoints/model_500000_steps.zip --timesteps 500000

Watch a trained agent (MP4 in `<run>/videos/`, viewable in Jupyter):

    python /opt/simrl/enjoy.py /workspace/runs/<run>/best/best_model.zip --episodes 3

CleanRL-style PPO (one readable file; copy is in `/workspace/sim-rl/` for editing):

    python /workspace/sim-rl/ppo_continuous.py --env Walker2d-v5 --total-timesteps 1000000 --capture-video

TensorBoard: open port 6006 from the pod's Connect menu. It reads `/workspace/runs`.

Long runs: start them so they survive closing the browser, e.g.

    nohup python /opt/simrl/train_sb3.py --env Humanoid-v5 --algo sac --timesteps 3000000 \
      > /workspace/runs/humanoid.log 2>&1 &

Your own environments: register a Gymnasium env (or `pip install` a package that
does) and pass its id with `--env`. Install extra packages with
`pip install <pkg>` (the image sets `PIP_BREAK_SYSTEM_PACKAGES=1`); for packages that
must survive a pod reset, add them to `requirements.txt` and rebuild.

## Storage and cost

- `/workspace` (volume) keeps runs, models, videos and your edited copies in
  `/workspace/sim-rl/` across pod stops. Everything else resets with the container.
- `/opt/simrl` always holds the image's original scripts; `/workspace/sim-rl` is seeded
  once and never overwritten by later image versions.
- A pod keeps billing until it is stopped; a stopped pod still bills for its volume.
  Terminate it when done after copying anything you need.
