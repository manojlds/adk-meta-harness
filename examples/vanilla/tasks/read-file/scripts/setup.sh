#!/usr/bin/env bash
set -euo pipefail

# Example setup hook for task execution.
# Fixtures are already copied into WORK_DIR by the executor.
# This hook writes a marker file to show setup ran.
mkdir -p "$LOGS_DIR"
printf 'setup ok\n' > "$LOGS_DIR/setup.log"
