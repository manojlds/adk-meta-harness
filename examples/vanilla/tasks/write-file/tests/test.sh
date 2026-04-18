#!/bin/bash
set -e

LOGS_DIR="${LOGS_DIR:-/logs}"
REWARD_DIR="${REWARD_DIR:-$LOGS_DIR/verifier}"
WORK_DIR="${WORK_DIR:-/app}"
OUTPUT_FILE="${OUTPUT_FILE:-$WORK_DIR/output.txt}"
mkdir -p "$REWARD_DIR"

# Check if output.txt exists with the expected content
if [ -f "$OUTPUT_FILE" ]; then
    CONTENT="$(cat "$OUTPUT_FILE" | tr -d '[:space:]')"
    EXPECTED="Thequickbrownfoxjumpsoverthelazydog"
    if [ "$CONTENT" = "$EXPECTED" ]; then
        echo "1.0" > "$REWARD_DIR/reward.txt"
    else
        echo "0.0" > "$REWARD_DIR/reward.txt"
    fi
else
    echo "0.0" > "$REWARD_DIR/reward.txt"
fi
