# adk-meta-harness

Meta-harness optimization for Google ADK agents. Automatically improves your
ADK agent's harness — system prompts, skills, tools, callbacks, and routing —
through an iterative propose-evaluate-gate loop.

## How it works

Following the [Meta-Harness](https://arxiv.org/abs/2603.28052) paper:

1. **Propose** — A coding agent CLI (OpenCode, Pi) reads the filesystem of all
   prior candidates, traces, and scores, then proposes one targeted harness change.
2. **Evaluate** — The proposed harness runs on Harbor tasks (search + holdout sets).
3. **Gate** — Keep only if holdout improves. Discard otherwise.
4. **Learn** — Accumulated insights are written to `learnings.md` for the next iteration.
5. **Repeat** — Continue the loop.

The key insight: the proposer has **full filesystem access** to all prior candidates,
execution traces, and scores — up to 10M tokens of diagnostic context per step.
This enables counterfactual diagnosis rather than guessing from a score.

## Install

```bash
pip install adk-meta-harness
```

## Quick start

```bash
# Run the optimization loop
adk-meta-harness optimize \
  --dataset path/to/harbor/tasks \
  --initial-harness path/to/harness \
  --proposer opencode \
  --iterations 10

# Evaluate a single harness
adk-meta-harness eval \
  --candidate path/to/harness \
  --dataset path/to/harbor/tasks
```

## What gets optimized

The proposer can modify any of these harness surfaces:

| Surface | File | ADK mapping |
|---|---|---|
| System prompt | `system_prompt.md` | `Agent(instruction=...)` |
| Skills | `skills/*/SKILL.md` | Agent Skills (agentskills.io spec) |
| Tools | `tools/*.py` | `Agent(tools=[...])` |
| Callbacks | `callbacks/*.py` | `before_model_callback`, `after_tool_callback` |
| Routing | `routing/*.yaml` | Multi-agent transfer rules |
| Agent config | `config.yaml` | Model, max_turns, stop_conditions |
| Full harness | `agent.py` | Top-level Agent construction |

Skills follow the [agentskills.io](https://agentskills.io) specification, with
full support for `SKILL.md` + `scripts/` + `references/` + `assets/`. Integration
with [adk-skills](https://github.com/manojlds/adk-skills) means the proposer can
create, modify, compose, and delete skills — not just edit text within them.

## Proposers

The proposer is pluggable. Currently supported:

- **OpenCode** — `--proposer opencode`
- **Pi** — `--proposer pi`
- **Any CLI** — `--proposer custom-cli-command`

The coding agent CLI operates on the harness directory as a filesystem, reading
traces and scores from prior candidates, and making targeted edits.

## Skills integration

adk-meta-harness uses [adk-skills](https://github.com/manojlds/adk-skills) for
skill discovery and activation within each candidate harness. The proposer can:

- **Add** a skill: Create a new skill directory with SKILL.md
- **Modify** a skill: Edit instructions, add failure patterns to references/
- **Compose** skills: Edit SKILL.md to reference other skills
- **Remove** a skill: Delete a skill directory that's hurting performance
- **Tune activation**: Modify tool descriptions or prompt injection format

## Tool search integration

[adk-tool-search](https://github.com/manojlds/adk-tool-search) provides dynamic
BM25-based tool discovery. The proposer can modify tool registration, search
parameters, and callback behavior as part of the harness surface.

## Architecture

```
adk-meta-harness/
├── src/adk_meta_harness/
│   ├── cli.py              # CLI: optimize, eval
│   ├── outer_loop.py       # Propose → Evaluate → Gate → Repeat
│   ├── candidate.py        # Candidate harness representation
│   ├── gate.py             # Holdout evaluation + keep/discard
│   ├── judge.py            # LLM judge for unlabeled traces
│   ├── learnings.py        # learnings.md accumulator
│   ├── harbor_adapter.py   # Harbor ADK agent runner
│   └── proposer/
│       ├── base.py          # ProposerProtocol
│       ├── coding_agent_cli.py  # Generic CLI adapter
│       ├── opencode.py     # OpenCode proposer
│       └── pi.py           # Pi proposer
├── configs/                # Starter harness configs
├── examples/               # Use-case examples
└── PROPOSER.md             # Template proposer instructions
```

## Related work

- [Meta-Harness](https://arxiv.org/abs/2603.28052) — The paper this is based on
- [canvas-org/meta-agent](https://github.com/canvas-org/meta-agent) — Open-source Meta-Harness for Claude SDK
- [kevinrgu/autoagent](https://github.com/kevinrgu/autoagent) — Autonomous agent engineering with Harbor
- [GEPA](https://github.com/gepa-ai/gepa) — Reflective prompt evolution (complementary inner loop)
- [Harbor](https://github.com/harbor-framework/harbor) — Agent evaluation framework
- [agentskills.io](https://agentskills.io) — Open standard for agent skills

## License

Apache 2.0