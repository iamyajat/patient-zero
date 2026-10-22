"""V2b: cascading infection with real tools, on AgentDojo suites wrapped in an orchestrator-worker hierarchy.

MultiAgentDojo (Peters-Gill et al. 2026) is described as released but has no public link, so this rebuilds its
structure on the `agentdojo` package (install with `uv sync --extra dojo`): an orchestrator delegates to sub-agents
through agent-as-tool calls, and each suite's tools are split among the sub-agents (split.py).
"""
