#!/bin/bash
set -e

LOGS_DIR="${LOGS_DIR:-/logs}"
REWARD_DIR="${REWARD_DIR:-$LOGS_DIR/verifier}"
AGENT_RESPONSE_FILE="${AGENT_RESPONSE_FILE:-$LOGS_DIR/agent/response.txt}"
mkdir -p "$REWARD_DIR"

RESPONSE="$(cat "$AGENT_RESPONSE_FILE" 2>/dev/null || echo '')"

# Check if the response mentions at least alpha.txt and beta.txt
PASS=0
if echo "$RESPONSE" | grep -qi "alpha"; then PASS=$((PASS+1)); fi
if echo "$RESPONSE" | grep -qi "beta"; then PASS=$((PASS+1)); fi

if [ "$PASS" -ge 2 ]; then
    echo "1.0" > "$REWARD_DIR/reward.txt"
elif [ "$PASS" -ge 1 ]; then
    echo "0.5" > "$REWARD_DIR/reward.txt"
else
    echo "0.0" > "$REWARD_DIR/reward.txt"
fi
