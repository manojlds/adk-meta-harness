#!/bin/bash
set -e

REWARD_DIR="/logs/verifier"
mkdir -p "$REWARD_DIR"

RESPONSE="$(cat /logs/agent/response.txt 2>/dev/null || echo '')"

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