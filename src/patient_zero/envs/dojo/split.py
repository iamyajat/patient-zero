"""How each AgentDojo suite's tools are split among sub-agents (following MultiAgentDojo's agent list)."""

from __future__ import annotations

# suite -> ordered (agent, name substrings); first match wins, the last agent takes the rest
SPLITS: dict[str, list[tuple[str, list[str]]]] = {
    "workspace": [("calendar_agent", ["calendar", "current_day"]), ("drive_agent", ["file"]), ("email_agent", [])],
    "banking": [("files_agent", ["read_file"]), ("account_agent", ["user_info", "password"]), ("transactions_agent", [])],
    "slack": [("web_agent", ["webpage"]), ("slack_agent", [])],
    "travel": [
        ("hotels_agent", ["hotel"]),
        ("restaurants_agent", ["restaurant"]),
        ("car_rental_agent", ["car"]),
        ("assistant_agent", []),
    ],
}


def split_tools(suite: str, tool_names: list[str]) -> dict[str, list[str]]:
    rules = SPLITS[suite]
    out: dict[str, list[str]] = {agent: [] for agent, _ in rules}
    for name in tool_names:
        agent = next((a for a, keys in rules if keys and any(k in name for k in keys)), rules[-1][0])
        out[agent].append(name)
    return {a: ns for a, ns in out.items() if ns}


def layout_agents(layout: str, suite: str, tool_names: list[str]) -> dict[str, list[str]]:
    """dojo_pair: one worker holds every tool. dojo_hub: specialised sub-agents."""
    if layout == "dojo_pair":
        return {"worker": list(tool_names)}
    if layout == "dojo_hub":
        return split_tools(suite, tool_names)
    raise ValueError(f"unknown dojo layout {layout!r}")
