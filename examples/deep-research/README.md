# Deep Research Example

This directory contains an example of using adk-meta-harness to optimize
a deep research agent.

This is an **example** — not part of the core library. The meta-harness is
use-case agnostic; you bring your own Harbor tasks and initial harness.

## Structure

- `initial_harness/` — Starter candidate harness with deep-research skill
  and dynamic tool search via adk-tool-search

## Usage

```bash
adk-meta-harness optimize \
  --dataset path/to/harbor/tasks \
  --initial-harness examples/deep-research/initial_harness \
  --proposer opencode \
  --iterations 10
```

The optimizer will iteratively improve the harness: adjusting the system prompt,
refining the deep-research skill instructions, adding/modifying tools, tuning
callbacks, and evolving the agent configuration.