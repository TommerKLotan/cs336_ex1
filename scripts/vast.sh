#!/usr/bin/env bash
# Set up / sync a vast.ai instance for training, and pull results back.
#
# Usage:
#   scripts/vast.sh setup [DATASET_NAME]   # copy code + data, install env (safe to re-run)
#   scripts/vast.sh pull                   # download models/ and logs/
#
# The instance's IP and port are read from vast.conf in the Onboarding directory:
#   VAST_IP=1.2.3.4
#   VAST_PORT=41234
# taken from its "Direct ssh connect" line on vast.ai, e.g. `ssh -p 41234 root@1.2.3.4`.
#
# If the host intercepts HTTPS (uv fails with "invalid peer certificate: UnknownIssuer")
# and you trust it anyway, run setup with TRUST_SYSTEM_CERTS=1 so uv uses the host's CA store.
set -euo pipefail

DEFAULT_DATASET="TinyStoriesV2-GPT4-train-bpe-2026-10-08_19-11-45-TinyStoriesV2-GPT4"
REMOTE_DIR="/workspace/cs336-ex1"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

CONFIG_FILE="$(cd "$LOCAL_DIR/.." && pwd)/vast.conf"

[[ $# -ge 1 ]] || usage
CMD="$1"
DATASET="${2:-$DEFAULT_DATASET}"

[[ -f "$CONFIG_FILE" ]] || { echo "Missing config file: $CONFIG_FILE" >&2; exit 1; }
# shellcheck source=/dev/null
source "$CONFIG_FILE"
IP="${VAST_IP:-}"
PORT="${VAST_PORT:-}"
[[ -n "$IP" && -n "$PORT" ]] || { echo "Set VAST_IP and VAST_PORT in $CONFIG_FILE" >&2; exit 1; }

# New vast hosts have unknown host keys; accept them on first connect.
SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
RSYNC=(rsync -az --info=progress2 -e "${SSH[*]}")
HOST="root@$IP"

setup() {
    local train="$LOCAL_DIR/tokenized_data/$DATASET-train.npy"
    local valid="$LOCAL_DIR/tokenized_data/$DATASET-valid.npy"
    for f in "$train" "$valid"; do
        [[ -f "$f" ]] || { echo "Missing dataset file: $f" >&2; exit 1; }
    done

    echo ">> Preparing remote (rsync, directories)"
    "${SSH[@]}" "$HOST" bash -s <<EOF
set -e
command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)
mkdir -p $REMOTE_DIR/tokenized_data
EOF

    echo ">> Syncing code and tokenizer"
    "${RSYNC[@]}" \
        --exclude .git --exclude .venv --exclude __pycache__ \
        --exclude .pytest_cache --exclude .ruff_cache \
        --exclude data --exclude models --exclude logs --exclude tokenized_data \
        --exclude '*.pdf' \
        "$LOCAL_DIR/" "$HOST:$REMOTE_DIR/"

    # --partial so an interrupted multi-GB upload resumes instead of restarting.
    echo ">> Syncing dataset $DATASET"
    "${RSYNC[@]}" --partial "$train" "$valid" "$HOST:$REMOTE_DIR/tokenized_data/"

    echo ">> Installing uv and Python environment"
    "${SSH[@]}" "$HOST" bash -s <<EOF
set -e
command -v uv >/dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
if [ "${TRUST_SYSTEM_CERTS:-0}" = 1 ]; then
    export UV_SYSTEM_CERTS=1
    grep -q UV_SYSTEM_CERTS ~/.bashrc || echo "export UV_SYSTEM_CERTS=1" >> ~/.bashrc
fi
cd $REMOTE_DIR
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available (host driver too old for this torch build?)'; print('GPU:', torch.cuda.get_device_name())"
EOF

    cat <<EOF

Done. Connect with:
  ssh -p $PORT $HOST -L 6006:localhost:6006
Then:
  cd $REMOTE_DIR
  uv run python cs336_basics/training_loop.py --dataset-name $DATASET ...
  uv run tensorboard --logdir logs --port 6006   # view at http://localhost:6006
EOF
}

pull() {
    echo ">> Pulling models/ and logs/"
    "${RSYNC[@]}" --partial "$HOST:$REMOTE_DIR/models" "$HOST:$REMOTE_DIR/logs" "$LOCAL_DIR/"
}

case "$CMD" in
    setup) setup ;;
    pull) pull ;;
    *) usage ;;
esac
