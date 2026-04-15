#!/bin/bash
set -e

REWARD_DIR="/logs/verifier"
mkdir -p "$REWARD_DIR"

# Check if the response contains the expected file content
RESPONSE="$(cat /logs/agent/response.txt 2>/dev/null || echo '')"

if echo "$RESPONSE" | grep -qi "hello world from the test file"; then
    echo "1.0" > "$REWARD_DIR/reward.txt"
else
    echo "0.0" > "$REWARD_DIR/reward.txt"
fi