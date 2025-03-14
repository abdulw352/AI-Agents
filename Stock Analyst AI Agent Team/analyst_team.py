import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import tool
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Any, Optional
import os
import datetime
import json
import yfinance as yf
import pandas as pd
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# Set up the Ollama model
llm = OllamaLLM(model="llama3.1")

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for information about the given query."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")['Close'].iloc[-1]
        return f"The current price of {ticker} is ${price:.2f}"
    except Exception as e:
        return f"Error retrieving stock price for {ticker}: {str(e)}"

@tool
def get_company_info(ticker: str) -> str:
    """Get general information about a company by its ticker symbol."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract the most relevant information
        company_data = {
            "Name": info.get("longName", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Market Cap": f"${info.get('marketCap', 0)/1000000000:.2f}B",
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "52 Week High": f"${info.get('fiftyTwoWeekHigh', 0):.2f}",
            "52 Week Low": f"${info.get('fiftyTwoWeekLow', 0):.2f}",
            "Description": info.get("longBusinessSummary", "N/A")
        }
        
        # Convert to a readable string format
        result = "### Company Information\n\n"
        for key, value in company_data.items():
            if key == "Description":
                result += f"\n**{key}**:\n{value}\n"
            else:
                result += f"**{key}**: {value}\n"
                
        return result
    except Exception as e:
        return f"Error retrieving company information for {ticker}: {str(e)}"

@tool
def get_analyst_recommendations(ticker: str) -> str:
    """Get analyst recommendations for a company."""
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        
        if recommendations is None or recommendations.empty:
            return f"No analyst recommendations found for {ticker}."
        
        # Get the most recent recommendations (last 5)
        recent_recommendations = recommendations.tail(5)
        
        # Convert to markdown table
        table = "### Recent Analyst Recommendations\n\n"
        table += "| Date | Firm | To Grade | From Grade | Action |\n"
        table += "|------|------|----------|------------|--------|\n"
        
        for idx, row in recent_recommendations.iterrows():
            date = idx.strftime("%Y-%m-%d")
            firm = row.get('Firm', 'N/A')
            to_grade = row.get('To Grade', 'N/A')
            from_grade = row.get('From Grade', 'N/A')
            action = row.get('Action', 'N/A')
            
            table += f"| {date} | {firm} | {to_grade} | {from_grade} | {action} |\n"
            
        return table
    except Exception as e:
        return f"Error retrieving analyst recommendations for {ticker}: {str(e)}"

@tool
def get_company_news(ticker: str) -> str:
    """Get recent news about a company."""
    try:
        # This is a simplification as yfinance doesn't have a direct news API
        # In a production environment, you might want to use a dedicated news API
        stock = yf.Ticker(ticker)
        company_name = stock.info.get('longName', ticker)
        
        # Use DuckDuckGo to search for recent news
        search = DuckDuckGoSearchRun()
        query = f"{company_name} recent news financial"
        results = search.run(query)
        
        return f"### Recent News for {company_name}\n\n{results}"
    except Exception as e:
        return f"Error retrieving news for {ticker}: {str(e)}"

# Create the Web Research Agent
web_tools = [search_web]
web_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Web Research Agent specialized in finding information on the internet.
Your role is to search the web for factual information about companies, markets, and financial news.
When responding, format information clearly and concisely. Focus on finding relevant, factual information.
Always cite your sources when possible.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""),
    ("human", "{input}"),
    ("human", "{agent_scratchpad}"),
])

web_agent = create_react_agent(
    llm=llm,
    tools=web_tools,
    prompt=web_agent_prompt
)

web_agent_executor = AgentExecutor(
    agent=web_agent,
    tools=web_tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Create the Finance Data Agent
finance_tools = [get_stock_price, get_company_info, get_analyst_recommendations, get_company_news]
finance_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Finance Data Agent specialized in retrieving and analyzing financial information.
Your role is to get accurate financial data about companies, stocks, and markets.
Always present numerical data in tables when appropriate.
Be precise with numbers and always specify the units (e.g., $, %, etc.).
Provide context and brief explanations with the data you present.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""),
    ("human", "{input}"),
    ("human", "{agent_scratchpad}"),
])

finance_agent = create_react_agent(
    llm=llm,
    tools=finance_tools,
    prompt=finance_agent_prompt
)

finance_agent_executor = AgentExecutor(
    agent=finance_agent,
    tools=finance_tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Define the Team Coordinator Agent
class AgentState(BaseModel):
    """State for the agent team."""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    next_agent: Optional[str] = None
    
def route_to_agent(state: AgentState) -> AgentState:
    """Route to the appropriate agent based on the query."""
    # Get the latest message
    last_message = state.messages[-1]["content"]
    
    # Determine which agent should handle this
    prompt = f"""
    Given this user query: "{last_message}"
    Which agent is better suited to answer?
    
    1. "web_agent" - For general web searches, news, information gathering
    2. "finance_agent" - For specific financial data, stock prices, company metrics
    
    Choose the best agent and only return "web_agent" or "finance_agent".
    """
    
    result = llm.invoke(prompt).strip().lower()
    
    # Update the state with the next agent
    if "finance" in result:
        state.next_agent = "finance_agent"
    else:
        state.next_agent = "web_agent"
    
    # Return the updated state
    return state

def call_web_agent(state: AgentState) -> AgentState:
    """Call the web research agent."""
    last_message = state.messages[-1]["content"]
    result = web_agent_executor.invoke({"input": last_message})
    response = result.get("output", "I couldn't find any relevant information.")
    
    state.messages.append({
        "role": "assistant",
        "content": f"[Web Agent] {response}",
        "agent": "web_agent"
    })
    state.next_agent = "coordinator"
    return state

def call_finance_agent(state: AgentState) -> AgentState:
    """Call the finance data agent."""
    last_message = state.messages[-1]["content"]
    result = finance_agent_executor.invoke({"input": last_message})
    response = result.get("output", "I couldn't retrieve the financial information requested.")
    
    state.messages.append({
        "role": "assistant",
        "content": f"[Finance Agent] {response}",
        "agent": "finance_agent"
    })
    state.next_agent = "coordinator"
    return state

def coordinator_agent(state: AgentState) -> AgentState:
    """Synthesize information from the specialized agents."""
    # Get the history of messages
    messages_history = [msg["content"] for msg in state.messages]
    
    # Create a simple prompt for the coordinator to synthesize the information
    coordinator_prompt = f"""
    Based on all the information gathered:
    
    {messages_history}
    
    Provide a comprehensive and cohesive response to the original user query.
    Make sure to integrate all relevant information from both web searches and financial data.
    Present financial data in tables when appropriate.
    """
    
    synthesis = llm.invoke(coordinator_prompt)
    
    state.messages.append({
        "role": "assistant",
        "content": synthesis,
        "agent": "coordinator"
    })
    
    # End the workflow
    return {"messages": state.messages, "next_agent": None}

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the workflow."""
    if state.next_agent is None:
        return END
    return state.next_agent

# Create the LangGraph workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router", route_to_agent)
workflow.add_node("web_agent", call_web_agent)
workflow.add_node("finance_agent", call_finance_agent)
workflow.add_node("coordinator", coordinator_agent)

# Add edges
workflow.add_conditional_edges(
    "router",
    should_continue,
    {
        "web_agent": "web_agent",
        "finance_agent": "finance_agent"
    }
)
workflow.add_edge("web_agent", "coordinator")
workflow.add_edge("finance_agent", "coordinator")
workflow.add_conditional_edges(
    "coordinator",
    should_continue,
    {
        "router": "router",
        END: END
    }
)

# Set entry point
workflow.set_entry_point("router")

# Compile the workflow
finance_team_graph = workflow.compile()

# Create Streamlit app
st.set_page_config(page_title="Stock Analyst AI Agent Team", page_icon=":chart_with_upwards_trend:")
st.title("Finance AI Agent Team")
st.markdown("""
This application uses a team of AI agents to provide financial analysis and information:
- **Web Research Agent**: Searches the web for information about companies and markets
- **Finance Data Agent**: Retrieves financial data like stock prices and company metrics
- **Coordinator**: Combines information from both agents to provide comprehensive answers
""")
# Add requirements information 
st.markdown("---")
st.toast("""
**Dependencies**:
Make sure you have Ollama installed and running with the llama3.1 model available.
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What financial information would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Researching...")
        
        # Run the agent workflow
        try:
            initial_state = AgentState(messages=[{"role": "user", "content": prompt}])
            result = finance_team_graph.invoke(initial_state)
            
            # Get the final response from the coordinator
            final_messages = result["messages"]
            coordinator_messages = [msg for msg in final_messages if msg.get("agent") == "coordinator"]
            
            if coordinator_messages:
                response = coordinator_messages[-1]["content"]
            else:
                response = "I'm sorry, I couldn't process your request properly."
                
            # Update placeholder with final response
            message_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            message_placeholder.markdown(f"An error occurred: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})

