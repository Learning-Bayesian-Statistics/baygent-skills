# baygent-skills

A set of skills to call your agent Bayes. Thomas Bayes.

[Agent Skills](https://agentskills.io) for Bayesian modeling, causal inference, and probabilistic thinking. Compatible with Claude Code, Kimi Code, Cursor, Gemini CLI, and any agent that supports the [Agent Skills spec](https://agentskills.io/specification).

## Available skills

| Skill | Description |
|---|---|
| [bayesian-workflow](bayesian-workflow/) | Full Bayesian modeling workflow with PyMC and ArviZ |

More skills coming soon. Issues and PRs are welcome!

## Quick install

### Claude Code

```bash
# Install a single skill
claude install-skill https://github.com/Learning-Bayesian-Statistics/baygent-skills/tree/main/bayesian-workflow
```

### Other compatible agents

Clone the repo and point your agent at the skill folder you need:

```bash
git clone https://github.com/Learning-Bayesian-Statistics/baygent-skills.git
```

Then copy the skill directory into your agent's skills location (e.g. `~/.config/agents/skills/` for Kimi Code).

## Philosophy

These skills are **opinionated and workflow-first**. They don't just teach an agent what PyMC functions exist — they enforce a specific sequence of steps (prior predictive checks, diagnostics, calibration, reporting) and guardrails (94% HDI, reproducible seeds, save-to-disk) that produce reliable analyses.

Each skill is focused and lean. Rather than one monolithic skill that covers everything, we build specialized skills that do one thing well:

- **bayesian-workflow** covers the fundamentals that every Bayesian analysis needs.

## About

Created by [Alexandre Andorra](https://alexandorra.github.io/), host of [Learning Bayesian Statistics](https://www.learnbayesstats.com/).

## License

MIT - see [LICENSE](LICENSE).
