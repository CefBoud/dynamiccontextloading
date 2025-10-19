# DynamicContextLoading for LLMs & MCP

## Overview
DynamicContextLoading is a technique for on-demand tool activation in LLMs and MCP, preventing context bloating by loading tools only when needed.

## Background
While developing [Moncoder](https://github.com/CefBoud/MonCoder/), a simple coding agent, I noticed the context was cluttered with tool definitions, many rarely used. This inefficiency inspired the `loader` tool which has brief summaries of tools and dynamically loads full details only when needed.

This approach suits MCP, where multiple servers can consume significant context space. Implement multi-level loading like cache hierarchies: server descriptions, tool summaries, and full definitions.

## Problem
LLMs have limited context windows. Including all tools in every interaction causes higher costs, slower responses, reduced accuracy, and token limit errors. This approach activates tools dynamically for a lean context.



## Example: Function Calling
See `dcl_function_calling.py` for implementation. Uses `litellm` (setup via `.env` based on `.env.example`).

**Sample Interaction:**
```
User: Calculate 15 * 7 and get weather in New York.

Activated tools: calculator, get_weather

15 * 7 = 105
Weather in New York: cloudy, 8°C.
```

## Example: MCP Integration
Integrates with MCP servers (e.g., GitHub via stdio) using multi-level loading (start with 1 and proceed with 2 and 3 only when needed):

1. **Server Descriptions**: Load high-level server info initially.
2. **Tool Summaries**: Load briefs for specific servers on demand.
3. **Full Tools**: Activate only needed tools for the task.

This progressive loading keeps context efficient.

Run `uv run dcl_mcp.py` (requires `.env` with `GITHUB_PERSONAL_ACCESS_TOKEN` for full demo). See the expanded example below for details.

<details>
<summary>Click to expand condensed example run (showing two-level loading sequence)</summary>

```bash
# Run the script (assumes .env with GITHUB_PERSONAL_ACCESS_TOKEN set)
uv run dcl_mcp.py

# Output shows MCP servers initializing and listing tools
load_mcp_tools github ['add_comment_to_pending_review', 'add_issue_comment', ...]
load_mcp_tools figma ['get_figma_data', 'download_figma_images']

# Initial loader description: Only server-level info loaded
generated loader tool description: Dynamic Tool Loader for managing MCP tools...
'github' MCP: The GitHub MCP server provides tools for comprehensive GitHub management...
'figma' MCP: This MCP server enables fetching detailed Figma file data...

# User query triggers loader activation
User: Provide a list of 5 of my public GitHub repositories

# LLM uses loader to load GitHub tool summaries (Level 2)
tool call: {"action":"load_tool_summaries","servers":["github"]}
Loader result: Loaded tool summaries for servers: github.

# Now context includes tool briefs for GitHub
Tools summaries loaded:
- get_me: Retrieves details of the authenticated GitHub user.
- search_repositories: Searches for repositories by name, topics, or metadata.
[... other tools ...]

# LLM activates specific tools (Level 3)
tool call: {"action":"load_tools","tools":["get_me","search_repositories"],"server":"github"}
Loader result: Activated tools from github: get_me, search_repositories.

# Tools now active and callable
active tools: loader, get_me, search_repositories

# LLM calls get_me to get user info
tool call: {} (for get_me)
get_me result: {"login":"CefBoud", ...}

# Then calls search_repositories with query
tool call: {"query":"user:CefBoud","sort":"updated"} (for search_repositories)
search_repositories result: List of 5 public repos (e.g., MonCoder, cefboud.github.io, etc.)

# Final response from LLM
Assistant: Here is a list of 5 of your public GitHub repositories...
```
</details>