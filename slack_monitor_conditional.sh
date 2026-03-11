#!/bin/bash
# Conditional LDM training monitor — Slack webhook notifications
# Features: start notification, periodic progress, error detection, completion
# Usage: Run in tmux after training starts: bash slack_monitor_conditional.sh

WEBHOOK_URL="${SLACK_WEBHOOK_URL:?Set SLACK_WEBHOOK_URL environment variable}"
EXPR_NAME="GarmageNet_6th_conditional_pc"
LOG_DIR="/home/rebuilderai/garmagenet-impl/log/${EXPR_NAME}"
PROGRESS_INTERVAL=3600  # Send progress update every 1 hour (seconds)

HOSTNAME=$(hostname)
START_TIME=$(date +%s)

echo "[monitor] Looking for LDM training process..."

# Find LDM training PID
TRAIN_PID=""
for i in $(seq 1 60); do
    TRAIN_PID=$(pgrep -f "python src/ldm.py.*${EXPR_NAME}" | head -1)
    if [ -n "$TRAIN_PID" ]; then
        break
    fi
    echo "[monitor] Waiting for LDM process... ($i/60)"
    sleep 5
done

if [ -z "$TRAIN_PID" ]; then
    echo "[monitor] LDM process not found after 5 minutes."
    curl -s -X POST -H "Content-type: application/json" \
        --data "{\"text\":\"Warning: *Conditional LDM Monitor*: Training process \`${EXPR_NAME}\` not found after 5 minutes.\"}" \
        "$WEBHOOK_URL"
    exit 1
fi

echo "[monitor] Tracking LDM training PID=$TRAIN_PID"

# Get GPU info
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)

# Send start notification
curl -s -X POST -H "Content-type: application/json" \
    --data "{\"text\":\"*Conditional LDM Training Started*\n- Experiment: \`${EXPR_NAME}\`\n- Type: Point Cloud Conditioning (POINT_E, 512-dim)\n- Method: FROM SCRATCH (no transfer learning)\n- PID: ${TRAIN_PID}\n- Server: ${HOSTNAME}\n- GPU: ${GPU_INFO}\n- Monitor: progress every 1h, auto-notify on error/completion\"}" \
    "$WEBHOOK_URL"

LAST_PROGRESS_TIME=$START_TIME

# Monitor loop
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - LAST_PROGRESS_TIME))

    # Periodic progress update
    if [ "$ELAPSED" -ge "$PROGRESS_INTERVAL" ]; then
        LAST_PROGRESS_TIME=$CURRENT_TIME

        # Get current epoch from wandb or log
        CURRENT_EPOCH=""
        if ls ${LOG_DIR}/wandb/latest-run/files/output.log 2>/dev/null; then
            CURRENT_EPOCH=$(grep -oP 'Epoch \d+' ${LOG_DIR}/wandb/latest-run/files/output.log 2>/dev/null | tail -1 | grep -oP '\d+')
        fi
        if [ -z "$CURRENT_EPOCH" ]; then
            CURRENT_EPOCH=$(grep -oP 'Epoch \d+' /tmp/cond_ldm_train.log 2>/dev/null | tail -1 | grep -oP '\d+')
        fi

        # Get GPU usage
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)

        # Get latest loss from wandb API (wandb-summary.json only exists after run ends)
        WANDB_RUN_ID=$(basename $(readlink -f ${LOG_DIR}/wandb/latest-run) | sed 's/run-[0-9_]*-//')
        LATEST_LOSS=$(/home/rebuilderai/miniconda3/bin/conda run -n garmagenet python3 -c "
import wandb, os
os.environ['WANDB_SILENT'] = 'true'
try:
    api = wandb.Api()
    run = api.run('GarmentGen/${WANDB_RUN_ID}')
    s = run.summary
    total = s.get('total_loss', None)
    latent = s.get('loss_latent', None)
    bbox = s.get('loss_bbox', None)
    if total is not None:
        print('%.4f (latent=%.4f, bbox=%.4f)' % (total, latent, bbox))
    else:
        print('n/a')
except:
    print('n/a')
" 2>/dev/null)

        # Calculate runtime
        RUNTIME_SEC=$((CURRENT_TIME - START_TIME))
        RUNTIME_H=$((RUNTIME_SEC / 3600))
        RUNTIME_M=$(((RUNTIME_SEC % 3600) / 60))

        MSG="*Conditional LDM Progress*\n"
        MSG+="- Epoch: ${CURRENT_EPOCH:-unknown} / 200,000\n"
        MSG+="- Loss: ${LATEST_LOSS:-n/a}\n"
        MSG+="- GPU: ${GPU_MEM} (${GPU_UTIL})\n"
        MSG+="- Runtime: ${RUNTIME_H}h ${RUNTIME_M}m"

        curl -s -X POST -H "Content-type: application/json" \
            --data "{\"text\":\"${MSG}\"}" \
            "$WEBHOOK_URL"
    fi

    # Check for NaN in recent log output
    if tail -100 /tmp/cond_ldm_train.log 2>/dev/null | grep -qiE "nan|NaN detected"; then
        CURRENT_EPOCH=$(grep -oP 'Epoch \d+' /tmp/cond_ldm_train.log 2>/dev/null | tail -1 | grep -oP '\d+')
        LAST_LINES=$(tail -10 /tmp/cond_ldm_train.log 2>/dev/null | sed 's/"/\\"/g' | tr '\n' ' ')
        curl -s -X POST -H "Content-type: application/json" \
            --data "{\"text\":\"*NaN Detected in Conditional LDM!*\n- Epoch: ${CURRENT_EPOCH:-unknown}\n- Last log: \`${LAST_LINES}\`\n- Action: Check emergency checkpoint\"}" \
            "$WEBHOOK_URL"
        # Don't exit — NaN detection code in trainer saves emergency ckpt and stops
    fi
done

echo "[monitor] Training process ended."
sleep 3

# Determine exit reason
CURRENT_EPOCH=$(grep -oP 'Epoch \d+' /tmp/cond_ldm_train.log 2>/dev/null | tail -1 | grep -oP '\d+')
RUNTIME_SEC=$(($(date +%s) - START_TIME))
RUNTIME_H=$((RUNTIME_SEC / 3600))
RUNTIME_M=$(((RUNTIME_SEC % 3600) / 60))

if tail -50 /tmp/cond_ldm_train.log 2>/dev/null | grep -qE "Traceback|Error|OOM|CUDA|RuntimeError"; then
    LAST_LINES=$(tail -15 /tmp/cond_ldm_train.log 2>/dev/null | sed 's/"/\\"/g' | tr '\n' ' ')
    MSG="*Conditional LDM Error!*\n- Epoch: ${CURRENT_EPOCH:-unknown}\n- Runtime: ${RUNTIME_H}h ${RUNTIME_M}m\n- Error: \`${LAST_LINES}\`"
elif [ "${CURRENT_EPOCH:-0}" -ge 200000 ]; then
    MSG="*Conditional LDM Complete!*\n- Final epoch: ${CURRENT_EPOCH}\n- Runtime: ${RUNTIME_H}h ${RUNTIME_M}m\n- Checkpoints: \`log/${EXPR_NAME}/\`"
else
    LAST_LINES=$(tail -5 /tmp/cond_ldm_train.log 2>/dev/null | sed 's/"/\\"/g' | tr '\n' ' ')
    MSG="*Conditional LDM Stopped*\n- Last epoch: ${CURRENT_EPOCH:-unknown}\n- Runtime: ${RUNTIME_H}h ${RUNTIME_M}m\n- Last log: \`${LAST_LINES}\`"
fi

curl -s -X POST -H "Content-type: application/json" \
    --data "{\"text\":\"${MSG}\"}" \
    "$WEBHOOK_URL"

echo "[monitor] Slack notification sent. Exiting."
