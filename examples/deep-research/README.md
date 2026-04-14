# Deep Research Example

ADK agent with deep-research skill and dynamic tool search via adk-tool-search.

This is an **example** — not part of the core library. The meta-harness is
use-case agnostic; you bring your own Harbor tasks and initial harness.

## Usage

```bash
amh optimize \
  --dataset path/to/harbor/tasks \
  --initial-harness examples/deep-research/initial_harness \
  --proposer opencode
```