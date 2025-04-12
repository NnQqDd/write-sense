"""Tools for the ReAct agent based on LangChain's tool system."""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults

# Check for Tavily API key
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    print("Warning: TAVILY_API_KEY environment variable is not set.")
    print("The web_search tool will use a placeholder implementation.")
    print("To use Tavily search, set the TAVILY_API_KEY environment variable.")
    
    # Placeholder implementation if Tavily API key is not set
    @tool
    def web_search(query: str) -> str:
        """Search the web for information on a given query.
        
        Args:
            query: The search query to look up on the web
            
        Returns:
            Search results as a string
        """
        # For demonstration purposes, return a mock response
        current_date = datetime.now().strftime("%Y-%m-%d")
        return f"Web search results for '{query}' (as of {current_date}):\n" + \
               f"This is a simulated response. In a real implementation, this would " + \
               f"return actual search results from a search engine API."
else:
    # Create the Tavily search tool if API key is available
    tavily_search = TavilySearchResults(
        max_results=5,          # Number of search results to return
        search_depth="advanced", # Use 'basic' or 'advanced' search
        include_answer=True,     # Include AI-generated answer in results
        include_images=False,    # Whether to include image results
        api_key=tavily_api_key   # Pass the API key
    )
    
    # Wrap the Tavily tool in our tool function
    @tool
    def web_search(query: str) -> str:
        """Search the web for information on a given query using Tavily.
        
        Args:
            query: The search query to look up on the web
            
        Returns:
            Search results as a string
        """
        try:
            # Call the Tavily search API
            results = tavily_search.invoke({"query": query})
            
            # Format the results
            formatted_results = f"Web search results for '{query}':\n\n"
            
            # Check if results is a list (direct results) or a dictionary with results key
            if isinstance(results, list):
                search_results = results
                # Check if AI answer is available in the first result
                ai_answer = None
                if search_results and len(search_results) > 0:
                    first_result = search_results[0]
                    if 'content' in first_result and len(first_result['content']) > 200:
                        # Extract a potential answer from the first content
                        ai_answer = first_result['content'].split('\n')[0]
            else:
                # Assume it's a dictionary with standard structure
                search_results = results.get("results", [])
                ai_answer = results.get("answer")
            
            # Add the AI-generated answer if available
            if ai_answer:
                formatted_results += f"Answer: {ai_answer}\n\n"
            
            # Add the search results with full context
            formatted_results += "Sources with full context:\n"
            for i, result in enumerate(search_results, 1):
                title = result.get('title', 'No title')
                url = result.get('url', 'No URL')
                content = result.get('content', '')
                
                formatted_results += f"{i}. {title}\n"
                formatted_results += f"   URL: {url}\n"
                
                # Include full content
                if content:
                    formatted_results += f"   Content:\n{content}\n"
                
                formatted_results += "\n"
            
            if not search_results:
                formatted_results += "No search results found. Try refining your query.\n"
            
            return formatted_results
            
        except Exception as e:
            return f"Error performing web search: {str(e)}"

@tool
def vector_search(query: str, limit: int = 5) -> str:
    """Search for information in a vector database.
    
    Args:
        query: The search query to look up in the vector database
        limit: Maximum number of results to return
        
    Returns:
        Search results as a string
    """
    # This is a placeholder. In a real implementation, you would:
    # 1. Connect to your vector database (e.g., FAISS, Pinecone, etc.)
    # 2. Convert the query to an embedding
    # 3. Perform similarity search
    # 4. Format and return the results
    
    # For demonstration purposes, return a mock response
    return f"Vector database search results for '{query}' (limit: {limit}):\n" + \
           f"This is a simulated response. In a real implementation, this would " + \
           f"return actual results from a vector database using semantic similarity search."

@tool
def calculator(expression: str) -> str:
    """Perform simple arithmetic calculations.
    
    Args:
        expression: The arithmetic expression to evaluate (e.g., '2 + 2')
        
    Returns:
        The result of the evaluation
    """
    try:
        # WARNING: eval is used here for demonstration purposes only
        # In a production environment, use a safer alternative
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Get all tools
def get_tools(include_calculator: bool = False) -> List:
    """Get the list of available tools.
    
    Args:
        include_calculator: Whether to include the calculator tool
        
    Returns:
        List of tools
    """
    tools = [web_search, vector_search]
    if include_calculator:
        tools.append(calculator)
    return tools 