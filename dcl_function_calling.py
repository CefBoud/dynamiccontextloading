"""
Example implementation of DynamicContextLoader for LLMs.

This module demonstrates how to create a generalized tool loader that:
1. Maintains a registry of tools
2. Generates brief descriptions dynamically using LiteLLM based on full tool definitions
3. Allows on-demand activation of tools for LLM context
"""

import json
from litellm import completion

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


class Tool:
    """Represents a tool with its definition and execution function."""

    def __init__(self, definition, function):
        self.definition = definition
        self.function = function
        # Register the tool by its name
        TOOL_REGISTRY[self.definition["function"]["name"]] = self


def generate_briefs(tools_list) -> dict:
    """
    Generate brief descriptions for a list of tools using LiteLLM.

    Args:
        tools_list (list): List of tool names to generate briefs for.

    Returns:
        dict: Dictionary mapping tool names to their generated descriptions.
    """
    if not tools_list:
        return {}

    # Collect full tool definitions
    tool_definitions = []
    for tool_name in tools_list:
        tool = TOOL_REGISTRY.get(tool_name)
        if tool:
            tool_definitions.append(tool.definition)

    if not tool_definitions:
        return {}

    # Create a nice prompt for the model to generate briefs
    tools_json = json.dumps(tool_definitions, indent=2)
    prompt = f"""You are a helpful assistant that generates concise, accurate briefs for tools based on their full definitions.

Below is a list of tool definitions in JSON format:

{tools_json}

Please generate a brief description for each tool. Each brief should be:
- Concise (1-2 sentences, under 100 characters)
- Accurate to the tool's purpose
- Helpful for users to understand what the tool does

Return a JSON object where keys are tool names and values are the brief descriptions.

Example output:
{{
  "tool1": "Brief description here.",
  "tool2": "Another brief description."
}}

Generate the briefs now:"""

    response = completion(
        messages=[{"role": "user", "content": prompt}],
        **llm_config,
    )
    # Handle LiteLLM response structure
    try:
        result_text = response.choices[0].message.content
        if result_text:
            # Parse the JSON response
            briefs = json.loads(result_text.strip())
            return briefs
    except Exception as e:
        # Fallback to simple truncation if LLM fails
        briefs = {}
        for tool_name in tools_list:
            tool = TOOL_REGISTRY.get(tool_name)
            if tool:
                desc = tool.definition.get("function", {}).get("description", "")
                briefs[tool_name] = desc[:100] + "..." if len(desc) > 100 else desc
        return briefs


def generate_loader_description(briefs):
    """Generate the description for the loader tool based on current briefs."""
    available_tools = "\n".join(f"- {name}: {desc}" for name, desc in briefs.items())
    return f"""Dynamic Tool Loader for managing available tools. Provides brief descriptions of each tool and allows activating multiple tools by name.

Available tools:
{available_tools}

Choosing tools will make them available in the model context for the next interaction.

Usage: Provide a list of exact tool names you want to activate."""


def create_loader_tool():
    """Create the DynamicContextLoader tool that manages dynamic tool activation."""

    # Get current tools list (this would be dynamic in a real implementation)
    current_tools = list(TOOL_REGISTRY.keys())
    briefs = generate_briefs(current_tools)
    description = generate_loader_description(briefs)
    print(f"generated loader tool description: {description}")

    def loader_execute(tool_names: list) -> str:
        """Execute the loader to activate specified tools."""
        if not isinstance(tool_names, list):
            return "Error: tool_names must be a list of strings."
        activated = []
        failed = []
        errors = []
        active_tool_names = [tool["function"]["name"] for tool in active_tools]

        for name in tool_names:
            if name in active_tool_names:
                errors.append(f"{name} already loaded.")
                continue
            if name not in briefs:
                failed.append(name)
                continue
            tool = TOOL_REGISTRY.get(name)
            if tool:
                active_tools.append(tool.definition)
                activated.append(name)
            else:
                failed.append(name)
        response = (
            f"Activated tools: {', '.join(activated)}."
            if activated
            else "No tools activated."
        )
        if failed:
            response += f" Failed to activate: {', '.join(failed)}."
        response += " They are now available for use in the next message."
        if errors:
            response += f"Errors: {', '.join(errors)}"
        return response

    LOADER_TOOL_DEFINITION = {
        "type": "function",
        "function": {
            "name": "loader",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tool names to activate and add to context.",
                    },
                },
                "required": ["tool_names"],
            },
        },
    }

    return Tool(definition=LOADER_TOOL_DEFINITION, function=loader_execute)


# Example tools for demonstration
def example_bash_tool():
    """Example bash tool."""

    async def bash_execute(command: str) -> str:
        import subprocess

        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.stdout + result.stderr
        except Exception as e:
            return str(e)

    BASH_DEFINITION = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Executes bash commands with security checks and timeout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    }
    return Tool(definition=BASH_DEFINITION, function=bash_execute)


def example_read_tool():
    """Example read file tool."""

    def read_execute(file_path: str) -> str:
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            return str(e)

    READ_DEFINITION = {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Reads files from the local filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read.",
                    },
                },
                "required": ["file_path"],
            },
        },
    }
    return Tool(definition=READ_DEFINITION, function=read_execute)


def example_calculator_tool():
    """Example calculator tool."""

    def calculate_execute(expression: str) -> str:
        try:
            # Simple eval for demo (not safe for production)
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    CALC_DEFINITION = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Performs mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2').",
                    },
                },
                "required": ["expression"],
            },
        },
    }
    return Tool(definition=CALC_DEFINITION, function=calculate_execute)


def example_weather_tool():
    """Example weather tool."""

    def get_weather_execute(location: str) -> str:
        # Mock weather API call
        import random

        conditions = ["sunny", "cloudy", "rainy", "snowy"]
        temp = random.randint(0, 30)
        return f"Weather in {location}: {random.choice(conditions)}, {temp}°C"

    WEATHER_DEFINITION = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gets the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for (e.g., 'New York').",
                    },
                },
                "required": ["location"],
            },
        },
    }
    return Tool(definition=WEATHER_DEFINITION, function=get_weather_execute)


# Initialize example tools
example_bash_tool()
example_read_tool()
example_calculator_tool()
example_weather_tool()

# Create the loader tool
loader_tool = create_loader_tool()


def main():
    """Coherent conversation example in one big loop."""
    user_prompt = "I need to calculate 15 * 7 and then get the weather in New York."
    print(f"{COLOR_USER}User: {user_prompt}{RESET}\n")
    messages = [
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    active_tools.append(loader_tool.definition)  # Start with just loader
    while True:
        # print(f"messages: {messages}")
        print(f"active tools: {[tool['function']['name'] for tool in active_tools]}")
        response = completion(
            messages=messages,
            tools=active_tools,
            tool_choice="auto",
            **llm_config,
        )
        # Handle response
        try:
            message = response.choices[0].message
        except AttributeError:
            print("Error in response handling")
            break
        print(f"response {response}")
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = tool_call.function.arguments
                if func_name == "loader":
                    # Activate tools
                    tool_names = json.loads(args).get("tool_names", [])
                    result = loader_tool.function(tool_names)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
                    print(f"{COLOR_LOADER}Activated tools: {tool_names}{RESET}")
                    print(f"{COLOR_DEBUG}Loader result: {result}{RESET}\n")
                else:
                    tool = TOOL_REGISTRY.get(func_name)
                    if tool:
                        result = tool.function(**json.loads(args))
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


if __name__ == "__main__":
    main()
