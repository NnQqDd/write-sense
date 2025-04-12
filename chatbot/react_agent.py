"""
ReAct Agent implementation using LangGraph and LangChain.
"""

import os
from typing import List, Dict, Any, Optional
import dotenv

dotenv.load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from tools import get_tools

class ReActAgent:
    """
    A ReAct agent that uses LangGraph's prebuilt components to process user requests,
    reason about them, use tools, and respond.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        include_calculator: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the ReAct agent with LangGraph components.
        
        Args:
            model_name: The name of the LLM model to use.
            include_calculator: Whether to include the calculator tool.
            verbose: Whether to print verbose output during execution.
        """
        # Check for OpenAI API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            api_key=self.api_key
        )
        
        # Get the tools
        self.tools = get_tools(include_calculator=include_calculator)
        
        # Create a system prompt
        system_message = (
            "You are a helpful AI assistant that can use tools to answer the user's questions. "
            "Answer directly without using tools when the user is making casual conversation."
        )
        
        # Bind the system message to the LLM
        llm_with_system = self.llm.bind(messages=[SystemMessage(content=system_message)])
        
        # Create the agent executor
        self.agent_executor = create_react_agent(
            llm_with_system,
            self.tools
        )
        
        # Initialize conversation history
        self.message_history = []
        
        # Set verbosity
        self.verbose = verbose
        
        # Print initialization information
        available_tools = ", ".join([tool.name for tool in self.tools])
        print(f"ReAct Agent initialized with model: {model_name}")
        print(f"Available tools: {available_tools}")
    
    def process_user_input(self, user_input: str) -> str:
        """
        Process user input and return a response.
        
        Args:
            user_input: The user's input/question.
            
        Returns:
            The agent's response as a string.
        """
        # Add the user message to history
        user_message = HumanMessage(content=user_input)
        self.message_history.append(user_message)
        
        # Run the agent
        if self.verbose:
            print(f"\nProcessing user input: '{user_input}'")
            print("Starting agent execution...")
            
            # Stream the agent's thinking process
            for event in self.agent_executor.stream(
                {"messages": self.message_history},
                stream_mode="values"
            ):
                # Print each step in the agent's reasoning
                latest_message = event["messages"][-1]
                if self.verbose:
                    latest_message.pretty_print()
                
            # Get the final result
            result = latest_message.content
            
        else:
            # Run without streaming for non-verbose mode
            response = self.agent_executor.invoke({"messages": self.message_history})
            self.message_history = response["messages"]
            result = self.message_history[-1].content
            
        # Return the agent's response
        return result
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.message_history = []
        print("Conversation history has been reset.")

# Example usage
if __name__ == "__main__":
    # When running as a script directly, use relative import
    from tools import get_tools
    
    # Create the agent
    agent = ReActAgent(include_calculator=True, verbose=True)
    
    # Test with sample questions
    test_inputs = [
        # "What is the capital of France?",
        # "Can you search the web for information about machine learning?",
        # "What's 25 times 16?",
        "What is the current price of Bitcoin?"
    ]
    
    for test_input in test_inputs:
        print("\n" + "="*50)
        print(f"User: {test_input}")
        response = agent.process_user_input(test_input)
        print(f"Agent: {response}")
        print("="*50) 