"""
Example implementation of DynamicContextLoader for LLMs using MCP (Model Context Protocol).

This module demonstrates how to integrate MCP servers (e.g., GitHub MCP server) to:
1. Load tools from an MCP server via stdio
2. Generate brief descriptions dynamically using LiteLLM based on full tool definitions
3. Allow on-demand activation of tools for LLM context
4. Call tools dynamically through the MCP client
"""

import logging

# Configure the root logger (or a specific logger)
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


import os
import json
import asyncio
from contextlib import AsyncExitStack
from litellm import completion
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import llm_config

# ANSI color codes
COLOR_USER = "\033[94m"  # Blue
COLOR_ASSISTANT = "\033[92m"  # Green
COLOR_LOADER = "\033[93m"  # Yellow
COLOR_TOOL_RESULT = "\033[95m"  # Magenta
COLOR_DEBUG = "\033[96m"  # Cyan
RESET = "\033[0m"  # Reset

# Central registry for tools
TOOL_REGISTRY = {}

active_tools = []

# Registry for MCP servers
MCP_SERVERS = {
    "github": {
        "command": "docker",
        "args": [
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server",
            "stdio",
        ],
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get(
                "GITHUB_PERSONAL_ACCESS_TOKEN"
            )
        },
    },
    "figma": {
        "command": "npx",
        "args": ["-y", "figma-developer-mcp", "--stdio"],
        "env": {"FIGMA_OAUTH_TOKEN": os.environ.get("FIGMA_OAUTH_TOKEN")},
    },
}

# Registry for MCP server managers
MCP_SERVER_MANAGERS = {}

# Store generated briefs for servers
SERVER_BRIEFS = {}

LOADER_STATE = {
    "servers": {}  # e.g., {"github": {"descriptions_loaded": True, "summaries_loaded": False, "active_tools": []}}
}


class MCPServerManager:
    """Manages an individual MCP server, including its session, parameters, and lifecycle."""

    def __init__(self, server_name, server_config):
        self.server_name = server_name
        self.server_config = server_config
        self.server_params = None
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.tools = []

    async def initialize(self):
        """Initialize the MCP server connection."""
        try:
            # Set up stdio server parameters
            self.server_params = StdioServerParameters(
                command=self.server_config["command"],
                args=self.server_config["args"],
                env=self.server_config.get("env", {}),
            )

            stdio, write = await self.exit_stack.enter_async_context(
                stdio_client(self.server_params)
            )
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )

            await self.session.initialize()
        except Exception as e:
            print(f"Failed to initialize {self.server_name}: {e}")
            raise

    async def list_tools(self):
        """List tools from this server."""
        if not self.session:
            raise RuntimeError(f"Server {self.server_name} not initialized")
        tools_result = await self.session.list_tools()
        self.tools = tools_result.tools
        for tool in self.tools:
            func = self.create_tool_function(tool.name)
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                    if hasattr(tool, "inputSchema")
                    else {},
                },
            }
            Tool(definition=tool_def, function=func, server=self.server_name)
        return self.tools

    async def close(self):
        """Close the server's exit stack."""
        await self.exit_stack.aclose()

    def create_tool_function(self, tool_name):
        """Create a function to execute a tool on this server."""

        async def tool_execute(**kwargs):
            if not self.session:
                raise RuntimeError(f"Server {self.server_name} not initialized")
            result = await self.session.call_tool(tool_name, arguments=kwargs)
            return str(result)

        return tool_execute


class Tool:
    """Represents a tool with its definition, execution function, and server."""

    def __init__(self, definition, function, server=None):
        self.definition = definition
        self.function = function
        self.server = server
        # Register the tool by its name with server info
        TOOL_REGISTRY[self.definition["function"]["name"]] = self


async def load_mcp_tools():
    """Load server descriptions from multiple MCP servers using MCPServerManager."""
    for server_name, server_config in MCP_SERVERS.items():
        try:
            # Create and initialize server manager
            manager = MCPServerManager(server_name, server_config)
            await manager.initialize()
            await manager.list_tools()
            MCP_SERVER_MANAGERS[server_name] = manager
            print(
                f"load_mcp_tools {server_name} {[tool.name for tool in manager.tools]} \n"
            )
            # Initialize state for this server
            LOADER_STATE["servers"][server_name] = {
                "descriptions_loaded": True,
                "summaries_loaded": False,
                "active_tools": [],
            }

            # Note: Tools are not loaded here; only server descriptions are set
        except Exception as e:
            print(f"Failed to load server description for {server_name}: {e}")
            continue


def generate_briefs(tools_list, mcp_server_name) -> dict:
    """
    Generate server summary and tool briefs for a given MCP server using LiteLLM.

    Args:
        tools_list (list): List of tool names for the server.
        mcp_server_name (str): Name of the MCP server.

    Returns:
        dict: {'server_summary': str, 'tool_briefs': dict}
    """
    if not tools_list or not mcp_server_name:
        return {"server_summary": "", "tool_briefs": {}}

    # Collect full tool definitions for the server
    # print(f"TOOL_REGISTRY {TOOL_REGISTRY}. tools_list {tools_list} ")
    tool_definitions = []
    for tool in tools_list:
        tool_info = TOOL_REGISTRY.get(tool.name)
        if tool_info and tool_info.server== mcp_server_name:
            tool_definitions.append(tool_info.definition)

    if not tool_definitions:
        return {"server_summary": "", "tool_briefs": {}}

    # Create prompt for both summaries
    tools_json = json.dumps(tool_definitions, indent=2)
    prompt = f"""You are a helpful assistant that generates concise summaries for MCP servers and their tools.

Below is a list of tool definitions for the {mcp_server_name} MCP server:

{tools_json}

Please provide:
1. A 1-2 sentence summary of what the server's tools can do overall (under 200 characters).
2. Brief descriptions for each tool (1-2 sentences, under 100 characters each).

Return a JSON object:
{{
  "server_summary": "Overall summary here.",
  "tool_briefs": {{
    "tool1": "Brief description.",
    "tool2": "Another brief."
  }}
}}

Generate now:"""

    response = completion(
        messages=[{"role": "user", "content": prompt}],
        **llm_config,
    )
    # Handle response
    try:
        if hasattr(response, "choices") and response.choices:
            result_text = response.choices[0].message.content
            if result_text:
                result = json.loads(result_text.strip())
                return result
    except Exception as e:
        pass
    # Fallback
    server_summary = (
        f"{mcp_server_name} MCP server provides various tools for specific tasks."
    )
    tool_briefs = {}
    for tool_name in tools_list:
        tool_info = TOOL_REGISTRY.get(tool_name)
        if tool_info:
            desc = tool_info.definition.get("function", {}).get("description", "")
            tool_briefs[tool_name] = desc[:100] + "..." if len(desc) > 100 else desc
    return {"server_summary": server_summary, "tool_briefs": tool_briefs}


def generate_loader_description():
    servers = []
    for server, data in SERVER_BRIEFS.items():
        server_summary = data.get("server_summary", "")
        tool_briefs = data.get("tool_briefs", {})
        server_loader_info = f"'{server}' MCP: {server_summary}"
        if LOADER_STATE["servers"][server]["summaries_loaded"]:
            server_loader_info += f"\n Tools summaries loaded:\n  {'\n'.join(f'- {name}: {desc}' for name, desc in tool_briefs.items())}"
        servers.append(server_loader_info)
    return f"""Dynamic Tool Loader for managing MCP tools. Provides descriptions and enables activation of multiple tools by name.

{"\n\n".join(servers)}

Tools become available in the model context for the next interaction.

Usage: Use 'load_tool_summaries' to load a server's tool summaries. Once loaded, use 'load_tools' to activate specific tools. If interested in a server's tools, first load summaries, then activate needed ones."""


def refresh_loader_tool_description():
    for tool in active_tools:
        if tool["function"]["name"] == "loader":
            tool["function"]["description"] = generate_loader_description()


def create_loader_tool():
    """Create the DynamicContextLoader tool that manages dynamic tool activation."""

    # Generate briefs only for servers with loaded descriptions
    for server_name in LOADER_STATE["servers"]:
        if LOADER_STATE["servers"][server_name]["descriptions_loaded"]:
            SERVER_BRIEFS[server_name] = generate_briefs(
                MCP_SERVER_MANAGERS[server_name].tools, server_name
            )

    description = generate_loader_description()
    print(f"generated loader tool description: {description}")

    async def loader_execute(
        action: str, servers: list = None, tools: list = None, server: str = None
    ) -> str:
        """Execute the loader to load servers or tools."""
        if action == "load_tool_summaries":
            if not servers:
                return "Error: servers list is required for load_tool_summaries action."
            activated = []
            failed = []
            for s in servers:
                activated.append(s)
                if s not in LOADER_STATE["servers"]:
                    return f"Error: server {s} not recognized"
                LOADER_STATE["servers"][s]["summaries_loaded"] = True
            return (
                f"Loaded tool summaries for servers: {', '.join(activated)}."
                if activated
                else "No tool summaries loaded."
            )

        elif action == "load_tools":
            if not tools or not server:
                return (
                    "Error: tools list and server are required for load_tools action."
                )
            if (
                server not in LOADER_STATE["servers"]
                or not LOADER_STATE["servers"][server]["summaries_loaded"]
            ):
                return f"Error: Tool summaries for {server} must be loaded first."
            activated = []
            failed = []
            errors = []
            active_tool_names = [tool["function"]["name"] for tool in active_tools]
            manager = MCP_SERVER_MANAGERS.get(server)
            if not manager:
                return f"Error: No manager for server {server}."
            for name in tools:
                if name in active_tool_names:
                    errors.append(f"{name} already loaded.")
                    continue

                tool_info = TOOL_REGISTRY.get(name)
                if tool_info and tool_info.server == server:
                    active_tools.append(tool_info.definition)
                    LOADER_STATE["servers"][server]["active_tools"].append(name)
                    activated.append(name)
                else:
                    failed.append(name)
            response = (
                f"Activated tools from {server}: {', '.join(activated)}."
                if activated
                else f"No tools activated from {server}."
            )
            if failed:
                response += f" Failed: {', '.join(failed)}."
            if errors:
                response += f" Errors: {', '.join(errors)}"
            return response
        else:
            return "Error: Invalid action. Use 'load_tool_summaries' or 'load_tools'."

    LOADER_TOOL_DEFINITION = {
        "type": "function",
        "function": {
            "name": "loader",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["load_tool_summaries", "load_tools"],
                        "description": "Action to perform: load_tool_summaries or load_tools.",
                    },
                    "servers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of MCP server names to load (for load_tool_summaries).",
                    },
                    "tools": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tool names to activate (for load_tools).",
                    },
                    "server": {
                        "type": "string",
                        "description": "MCP server name for the tools (for load_tools).",
                    },
                },
                "required": ["action"],
            },
        },
    }

    return Tool(definition=LOADER_TOOL_DEFINITION, function=loader_execute)


async def main():
    """Coherent conversation example in one big loop with dynamic MCP tool loading."""
    # Initialize MCP and load tools
    await load_mcp_tools()

    # Create the loader tool
    loader_tool = create_loader_tool()

    # Start conversation
    user_prompt = "Provide a list of 5 of my public GitHub repositories"
    print(f"{COLOR_USER}User: {user_prompt}{RESET}\n")
    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    active_tools.append(loader_tool.definition)  # Start with just loader

    try:
        while True:
            
            print(f"active tools:")
            for tool in active_tools:
                  print(f"{tool['function']['name']} : {tool['function']['description']}")

            response = completion(
                messages=messages,
                tools=active_tools,
                tool_choice="auto",
                **llm_config,
            )

            # Handle response
            try:
                if hasattr(response, "choices") and response.choices:
                    message = response.choices[0].message
                else:
                    print("Error in response handling")
                    break
            except AttributeError:
                print("Error in response handling")
                break

            messages.append(message)

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    args = tool_call.function.arguments
                    print(f"tool call {tool_call}")
                    if func_name == "loader":
                        # Activate tools or servers
                        arguments = json.loads(args)
                        result = await loader_tool.function(**arguments)
                        refresh_loader_tool_description()  # some summaries might have been added
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            }
                        )
                        print(
                            f"{COLOR_LOADER}Loader action: {arguments.get('action', 'unknown')}{RESET}"
                        )
                        print(f"{COLOR_DEBUG}Loader result: {result}{RESET}\n")
                    else:
                        tool = TOOL_REGISTRY.get(func_name)
                        if tool:
                            # Call the async tool function
                            print(f"calling tool {tool}")
                            result = await tool.function(**json.loads(args))
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": result,
                                }
                            )
                            print(
                                f"{COLOR_TOOL_RESULT}{func_name} result: {result}{RESET}\n"
                            )
            else:
                print(f"{COLOR_ASSISTANT}Assistant: {message.content}{RESET}\n")
                break
    finally:
        # Close all server managers
        for manager in MCP_SERVER_MANAGERS.values():
            try:
                await manager.close()
            except:
                pass


if __name__ == "__main__":
    asyncio.run(main())
