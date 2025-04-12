# ReAct Agent with LangGraph

This module provides a ReAct (Reasoning + Acting) agent implementation using LangGraph and LangChain for processing natural language requests through reasoning, acting, and observing cycles.

## Overview

The ReAct approach combines:

- **Reasoning**: Step-by-step thinking about how to solve a task
- **Acting**: Taking actions using tools based on reasoning
- **Observing**: Processing results from tools to inform further reasoning

## Usage

```python
from chatbot.react_agent import ReActAgent

# Create the ReAct agent
agent = ReActAgent(
    model_name="gpt-4o-mini",     # Optional, defaults to gpt-4o-mini
    include_calculator=True,       # Optional, defaults to False
    verbose=True                   # Optional, defaults to False
)

# Process user input
response = agent.process_user_input("What's the capital of France?")
print(f"Agent response: {response}")

# Reset conversation history if needed
agent.reset_conversation()
```

## Available Tools

The agent comes with several built-in tools:

1. **web_search** - Searches the web for information using Tavily Search API (requires a Tavily API key)
2. **vector_search** - Searches a vector database for relevant information
3. **calculator** - (Optional) Performs simple arithmetic calculations

### Tavily Web Search Integration

The web_search tool uses the Tavily Search API, which is designed specifically for AI agents. To use it:

1. Sign up for a free account at [Tavily](https://tavily.com/) to get an API key
2. Set your API key as an environment variable:

   ```
   export TAVILY_API_KEY="your-tavily-api-key"
   ```

3. Or enter it when prompted by the example script

If a Tavily API key is not found, the tool will fall back to a placeholder implementation.

Features of the Tavily integration:

- Real-time, factual search results
- AI-generated answers to search queries
- Source URLs and snippets from web pages
- 1,000 free searches per month

## Creating Custom Tools

You can create custom tools using LangChain's tool decorator:

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(param1: str) -> str:
    """Description of what the tool does.
    
    Args:
        param1: Description of parameter
        
    Returns:
        The result
    """
    # Implement your tool logic here
    return f"Result of processing {param1}"

# Add your tools to the get_tools function in tools.py
def get_tools(include_calculator: bool = False):
    tools = [web_search, vector_search, my_custom_tool]
    if include_calculator:
        tools.append(calculator)
    return tools
```

## LangGraph Integration

This implementation uses LangGraph's `create_react_agent` function to create a streamlined ReAct agent that follows standard patterns for:

1. Reasoning about user requests
2. Deciding when to use tools vs. responding directly
3. Using tools to gather information
4. Formulating final responses based on reasoning and tool outputs

## Requirements

- OpenAI API key set as an environment variable (`OPENAI_API_KEY`)
- Tavily API key for web search (optional)
- LangChain libraries (`langchain-openai`, `langchain-core`, `langchain-community`)
- LangGraph library
- Tavily Python SDK (`tavily-python`)

## How It Works

The agent follows these steps for each user input:

1. Adds the user message to conversation history
2. Executes the LangGraph ReAct agent with the current conversation
3. The agent internally:
   - Reasons about how to respond to the user
   - Uses tools when needed to gather information
   - Formulates a response based on the tools' outputs
4. Returns the final response to the user

The implementation handles maintaining conversation history and can optionally show the agent's reasoning process when `verbose=True`.

## Example

See `example.py` for a complete interactive demo of using the ReAct agent.
