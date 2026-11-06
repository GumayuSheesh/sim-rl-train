# Maintainer notes — verification status

Recorded 2026-11-06. Package versions pinned in requirements.txt were tested
together with torch 2.8.0 (same torch as the base image) on Python 3.12.

| Check | Result |
|---|---|
| Project files authored | done |
| Python scripts compile; notebook passes nbformat validation; workflow YAML parses | passed |
| check_env.py on macOS arm64, CPU, MUJOCO_GL=glfw | PASS (physics ~22k steps/s, render, 2k-step PPO, save/load) |
| train_sb3.py: PPO + 4 subproc envs + VecNormalize, checkpoints, eval/best model, resume from checkpoint with stats; TQC and SAC short runs; no-overwrite run naming | passed (macOS, CPU) |
| enjoy.py: normalization stats lookup (checkpoint + final), MP4 recording | passed (macOS, CPU) |
| ppo_continuous.py: Hopper-v5, 102k steps, return 10 -> 822; TensorBoard scalars incl. episodic_return; RecordVideo | passed (macOS, CPU) |
| post_start.sh: seeding once, user edits kept, TensorBoard served HTTP 200 with run scalars, idempotent rerun, exits 0 when TensorBoard cannot start | passed on macOS with a fake workspace (`setsid` replaced, as macOS lacks it) |
| Docker image build | NOT run (no Docker on the authoring machine) |
| Linux/CUDA: torch on GPU, EGL rendering via NVIDIA, base `/start.sh` invoking `/post_start.sh` in tag 1.0.2 | UNVERIFIED |
| Runpod pod end-to-end (ports 8888/6006/22 via proxy, restart, volume retention) | UNVERIFIED |

Acceptance on the first real pod: `python /opt/simrl/check_env.py --require-gpu`
must PASS, port 6006 must show TensorBoard, Jupyter must open with
JUPYTER_PASSWORD, and after a pod restart `/workspace/sim-rl` edits must remain.
If `render` fails with EGL, try `MUJOCO_GL=osmesa` and record which host/GPU failed.

Known limits:
- mujoco 4.0.0 prints a `settotalmass` deprecation warning for some Gymnasium v5
  models; harmless.
- ppo_continuous.py saves agent weights only, not the NormalizeObservation stats.
