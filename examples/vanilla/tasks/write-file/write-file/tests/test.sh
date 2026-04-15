#!/bin/bash
set -e

REWARD_DIR="/logs/verifier"
mkdir -p "$REWARD_DIR"

# Check if output.txt exists with the expected content
if [ -f "/app/output.txt" ]; then
    CONTENT="$(cat /app/output.txt | tr -d '[:space:]')"
    EXPECTED="Thequickbrownfoxjumpsoverthelazydog"
    if [ "$CONTENT" = "$EXPECTED" ]; then
        echo "1.0" > "$REWARD_DIR/reward.txt"
    else
        echo "0.0" > "$REWARD_DIR/reward.txt"
    fi
else
    echo "0.0" > "$REWARD_DIR/reward.txt"
fi