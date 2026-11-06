#!/bin/bash
# Runs once per container start via the base image's /start.sh hook.
# The base /start.sh uses `set -e`, so this script must NEVER return non-zero:
# a failure here would kill the container and cause a restart loop.

RUNS_DIR="${RL_RUNS_DIR:-/workspace/runs}"
SIMRL_HOME="${SIMRL_HOME:-/opt/simrl}"
TB_PORT="${TENSORBOARD_PORT:-6006}"
WS="${WORKSPACE_DIR:-/workspace}"

{
    mkdir -p "$RUNS_DIR"

    # Seed an editable copy of the examples ONCE; never overwrite user edits.
    if [[ ! -e "$WS"/sim-rl ]]; then
        mkdir -p "$WS"/sim-rl
        cp -n "$SIMRL_HOME"/*.py "$SIMRL_HOME"/*.ipynb "$WS"/sim-rl/ 2>/dev/null
        echo "[post_start] seeded examples into $WS/sim-rl"
    else
        echo "[post_start] $WS/sim-rl exists; leaving it untouched"
    fi

    # TensorBoard in the background, bound to 0.0.0.0 for the Runpod proxy.
    if [[ "${DISABLE_TENSORBOARD:-false}" == "true" ]]; then
        echo "[post_start] TensorBoard disabled (DISABLE_TENSORBOARD=true)"
    elif pgrep -f "tensorboard.*--port[= ]$TB_PORT" >/dev/null; then
        echo "[post_start] TensorBoard already running on :$TB_PORT"
    else
        setsid nohup python -m tensorboard.main --logdir "$RUNS_DIR" \
            --host 0.0.0.0 --port "$TB_PORT" --reload_interval 15 \
            >/var/log/tensorboard.log 2>&1 < /dev/null &
        sleep 3
        if pgrep -f "tensorboard.*--port[= ]$TB_PORT" >/dev/null; then
            echo "[post_start] TensorBoard started on :$TB_PORT (logdir $RUNS_DIR)"
        else
            echo "[post_start] WARNING: TensorBoard failed to start; see /var/log/tensorboard.log" >&2
            tail -n 20 /var/log/tensorboard.log >&2
        fi
    fi
} || echo "[post_start] WARNING: a step failed; the pod keeps running" >&2

exit 0
