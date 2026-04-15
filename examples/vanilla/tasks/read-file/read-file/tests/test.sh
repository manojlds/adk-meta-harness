#!/bin/bash
set -e

LOGS_DIR="${LOGS_DIR:-/logs}"
REWARD_DIR="${REWARD_DIR:-$LOGS_DIR/verifier}"
AGENT_RESPONSE_FILE="${AGENT_RESPONSE_FILE:-$LOGS_DIR/agent/response.txt}"
mkdir -p "$REWARD_DIR"

# Check if the response contains the expected file content
RESPONSE="$(cat "$AGENT_RESPONSE_FILE" 2>/dev/null || echo '')"

if echo "$RESPONSE" | grep -qi "hello world from the test file"; then
    echo "1.0" > "$REWARD_DIR/reward.txt"
else
    echo "0.0" > "$REWARD_DIR/reward.txt"
fi
