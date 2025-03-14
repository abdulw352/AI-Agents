# Finance AI Agent Team

A multi-agent system for financial analysis and research powered by LangChain, LangGraph, and Ollama.


## Overview

This application implements an agent system focused on financial analysis and research. It combines web search capabilities with financial data retrieval to provide comprehensive answers to user queries about companies, stocks, markets, and financial trends.

The system uses a team of specialized agents orchestrated through LangGraph's workflow framework and leverages the power of Ollama's llama3.1 model for natural language understanding and generation.

## Features

- **Multi-Agent Architecture**: Specialized agents for web research and financial data retrieval
- **Intelligent Routing**: Automatically directs queries to the most appropriate specialized agent
- **Rich Financial Data**: Access to stock prices, company information, analyst recommendations, and news
- **Interactive Interface**: Clean Streamlit-based chat interface for easy interaction
- **Well-Structured Responses**: Presents financial data in formatted tables and structured reports
- **Synthesized Information**: Coordinator agent combines information from multiple sources for comprehensive answers

## Architecture

The system is built on a state-based workflow architecture with several key components:

### Agents

1. **Web Research Agent**: 
   - Specializes in finding information on the internet
   - Uses DuckDuckGo search for retrieving relevant web content
   - Focuses on factual information about companies, markets, and financial news

2. **Finance Data Agent**:
   - Retrieves and analyzes financial information
   - Accesses stock prices, company metrics, analyst recommendations, and news
   - Presents numerical data in well-formatted tables

3. **Coordinator Agent**:
   - Orchestrates the workflow and routes queries to specialized agents
   - Synthesizes information from multiple sources
   - Delivers cohesive, comprehensive responses to users

### Workflow

The system uses LangGraph to implement a state-based workflow:

1. **Query Analysis**: Determines which specialized agent should handle the user query
2. **Information Retrieval**: Routes the query to the appropriate agent(s) to gather information
3. **Synthesis**: Combines information from specialized agents into a comprehensive response
4. **Delivery**: Presents the final answer to the user in a structured, readable format

![Architecture Diagram](https://via.placeholder.com/800x400?text=Agent+Architecture+Diagram)

## Tools

The system implements several specialized tools:

- **Web Search**: Searches the internet for information about companies and financial topics
- **Stock Price**: Retrieves current and historical stock prices
- **Company Information**: Gets detailed information about companies including sector, market cap, and business description
- **Analyst Recommendations**: Retrieves recent analyst ratings and recommendations
- **Company News**: Finds recent news about companies and markets

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- llama3.1 model pulled into Ollama

### Installation

1. Clone this repository:
   ```bash
   git clone 
   cd "Stock Analyst AI Agent Team"
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have the llama3.1 model in Ollama:
   ```bash
   ollama pull llama3.1
   ```

### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at http://localhost:8501 in your web browser.

## Configuration

You can configure the system by modifying the following parameters in the code:

- **Model**: Change the Ollama model by modifying the `llm = OllamaLLM(model="llama3.1")` line
- **Tools**: Add or modify tools in the respective agent sections
- **Prompts**: Customize the system prompts for each agent to adjust their behavior

## Dependencies

The application requires the following main packages:

```
langchain==0.1.12
langchain-community==0.0.29
langchain-core==0.1.30
langchain-ollama==0.0.2
langgraph==0.0.36
streamlit==1.32.0
ollama==0.1.7
yfinance==0.2.36
pandas==2.1.4
pydantic==2.6.3
```

A complete `requirements.txt` file is included in the repository.

## Code Structure

- **analyst_team.py**: Main application file containing all agent definitions and the Streamlit interface
- **requirements.txt**: List of dependencies
- **README.md**: Documentation (this file)

### Key Code Sections

- **Tool Definitions**: Custom tools for web search and financial data retrieval
- **Agent Definitions**: Specialized agents with their respective tools and prompts
- **LangGraph Workflow**: State-based workflow definition and routing logic
- **Streamlit Interface**: User interface and interaction handling

## Usage Examples

### Researching a Company

User: "Tell me about Apple's recent financial performance and analyst outlook"

The system will:
1. Route the query to both the Web Research Agent and Finance Data Agent
2. Retrieve recent financial data, stock performance, and analyst recommendations
3. Search the web for recent news and analyst reports
4. Synthesize this information into a comprehensive answer

### Market Analysis

User: "How have tech stocks performed in the last month? Focus on FAANG companies."

The system will:
1. Use the Finance Data Agent to retrieve stock performance data
2. Use the Web Research Agent to find market analysis and trends
3. Combine the information into a cohesive market overview

## Limitations

- The system requires Ollama to be running locally with the llama3.1 model
- Financial data is retrieved from yfinance, which may have rate limits or occasional API changes
- Web search is performed through DuckDuckGo, which may not always provide the most comprehensive results
- The system does not persist data between sessions beyond the Streamlit chat history

## Future Improvements

- Add support for more financial data sources and APIs
- Implement persistent storage for conversation history and research findings
- Add visualization capabilities for financial data
- Integrate additional specialized agents for different domains (e.g., economic indicators, cryptocurrency)
- Implement user portfolio personalized research

## License

This project is licensed under the MIT License

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for providing the agent framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for the workflow orchestration capabilities
- [Ollama](https://ollama.ai/) for local LLM inference
- [yfinance](https://github.com/ranaroussi/yfinance) for financial data retrieval
- [Streamlit](https://streamlit.io/) for the interactive interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Added some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request