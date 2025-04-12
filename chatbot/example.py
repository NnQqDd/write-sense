#!/usr/bin/env python3
"""
Example script demonstrating how to use the LangGraph-based ReAct agent.
"""

import os
import getpass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the ReAct agent
from chatbot.react_agent import ReActAgent

def main():
    """
    Main function to demonstrate ReAct agent usage with LangGraph.
    """
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in environment variables.")
        print("Please enter your OpenAI API key (it won't be stored):")
        api_key = getpass.getpass()
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Check for Tavily API key (for web search)
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("\nTavily API key not found. Web search will use a placeholder implementation.")
        print("Would you like to set a Tavily API key for actual web search? (y/n)")
        response = input()
        if response.lower() in ["y", "yes"]:
            print("Please enter your Tavily API key (it won't be stored):")
            tavily_api_key = getpass.getpass()
            os.environ["TAVILY_API_KEY"] = tavily_api_key
            print("Tavily API key set. Web search will use the Tavily search API.")
        else:
            print("Continuing with placeholder web search implementation.")
    
    # Create the ReAct agent
    try:
        agent = ReActAgent(
            model_name="gpt-4o-mini",  # You can try different models
            include_calculator=True,    # Include the calculator tool
            verbose=True                # Print detailed execution steps
        )
        
        print("\nReAct Agent created successfully!")
        print("Type 'exit' to quit the demo.")
        print("Type 'reset' to reset the conversation history.")
        
        # Interactive demo
        while True:
            user_input = input("\nEnter your request: ")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting demo. Goodbye!")
                break
            
            if user_input.lower() == "reset":
                agent.reset_conversation()
                continue
            
            print("Processing your request...")
            response = agent.process_user_input(user_input)
            print(f"\nAgent response: {response}")
            
    except Exception as e:
        print(f"Error creating or using the ReAct agent: {e}")

if __name__ == "__main__":
    main() 