#!/usr/bin/env bash
# Set up / sync a vast.ai instance for training, and pull results back.
#
# Usage:
#   scripts/vast.sh setup <IP> <PORT> [DATASET_NAME]   # copy code + data, install env (safe to re-run)
#   scripts/vast.sh pull  <IP> <PORT>                  # download models/ and logs/
#
# IP and PORT come from the instance's "Direct ssh connect" line on vast.ai,
# e.g. `ssh -p 41234 root@1.2.3.4` -> IP=1.2.3.4 PORT=41234.
set -euo pipefail

DEFAULT_DATASET="TinyStoriesV2-GPT4-train-bpe-2026-10-08_19-11-45-TinyStoriesV2-GPT4"
REMOTE_DIR="/workspace/cs336-ex1"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

[[ $# -ge 3 ]] || usage
CMD="$1"
IP="$2"
PORT="$3"
DATASET="${4:-$DEFAULT_DATASET}"

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
