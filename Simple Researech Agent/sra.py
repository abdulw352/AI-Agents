from typing import TypedDict, List, Annotated
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM

class ResearchState(TypedDict):
    """Tracks the state of our research workflow using a TypedDict."""
    question: str
    initial_search: List[str]
    research_notes: List[str]
    final_summary: str

def search_web(state: ResearchState):
    """Perform initial web search based on research question."""
    search_tool = DuckDuckGoSearchResults(backend='auto')
    results = search_tool.run(state['question'])
    return {
        'initial_search': results.split('\n')[:3]  # Take top 3 results
    }

def analyze_results(state: ResearchState, model: BaseChatModel):
    """Analyze search results and generate research notes."""
    prompt = f"""
    Research Question: {state['question']}
    Search Results:
    {chr(10).join(state['initial_search'])}

    Synthesize these search results into structured research notes.
    Focus on key insights, main arguments, and interesting perspectives.
    """
    
    messages = [HumanMessage(content=prompt)]
    research_analysis = model.invoke(messages)
    return {
        'research_notes': [research_analysis]
    }

def generate_summary(state: ResearchState, model: BaseChatModel):
    """Create a comprehensive summary of research findings."""
    prompt = f"""
    Research Question: {state['question']}
    Research Notes:
    {chr(10).join(state['research_notes'])}

    Compose a clear, concise summary that answers the original research question.
    Include key points, insights, and provide a well-structured narrative.
    """
    
    messages = [HumanMessage(content=prompt)]
    final_summary = model.invoke(messages)
    return {
        'final_summary': final_summary
    }

def create_research_workflow():
    model = OllamaLLM(model="llama3.1")
    
    workflow = StateGraph(ResearchState)
    workflow.add_node("search", search_web)
    workflow.add_node("analyze", lambda state: analyze_results(state, model))
    workflow.add_node("summarize", lambda state: generate_summary(state, model))
    
    workflow.set_entry_point("search")
    workflow.add_edge("search", "analyze")
    workflow.add_edge("analyze", "summarize")
    workflow.add_edge("summarize", END)
    
    return workflow.compile()

# Example Usage
def run_research(question):
    research_agent = create_research_workflow()
    initial_state = {
        'question': question,
        'initial_search': [],
        'research_notes': [],
        'final_summary': ''
    }
    
    result = research_agent.invoke(initial_state)
    print(result['final_summary'])


if __name__ == "__main__":
    # Try it out!
    run_research("Impact of AI on modern software development")