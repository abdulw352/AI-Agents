import os
from typing import Dict, List, Optional, Any, Tuple
import operator
from functools import reduce

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import ChatLanguageModel
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

import langchain.schema as schema
from langchain.schema import BasePromptTemplate
import langgraph.graph as graph
from langgraph.graph import END, StateGraph
from langgraph.checkpoint import MemorySaver

# Define agent personalities
AGENT_PERSONALITIES = {
    "researcher": "You are a thorough researcher who values accuracy and depth. You focus on facts and details, always citing sources. You're methodical and exhaustive in your analysis.",
    
    "synthesizer": "You are a talented synthesizer who can distill complex information into clear, concise explanations. You excel at connecting ideas and making information accessible while maintaining accuracy.",
    
    "critic": "You are a critical thinker who questions assumptions and identifies weaknesses in arguments. You point out inconsistencies and logical fallacies, suggesting improvements. Your goal is to strengthen arguments through constructive criticism.",
    
    "creative": "You are a creative thinker who finds novel connections and fresh perspectives. You consider unusual angles and innovative solutions, making information engaging and memorable while staying true to the facts."
}

JUDGE_SYSTEM_PROMPT = """You are a fair and impartial judge. Your task is to evaluate responses from multiple agents and determine which response best answers the user's query.

Consider the following criteria:
1. Accuracy - Is the information correct and well-supported by evidence?
2. Relevance - Does the response directly address the user's query?
3. Completeness - Does the response cover all important aspects of the query?
4. Clarity - Is the response well-organized and easy to understand?
5. Evidence-based - Is the response backed by the search results provided?

You will receive:
- The original user query
- Search results from DuckDuckGo
- Responses from multiple agents
- Any voting or feedback provided by the agents

Based on these inputs, you must:
1. Evaluate each response
2. Identify strengths and weaknesses of each
3. Choose the best response OR suggest how to combine elements from multiple responses
4. If the responses have converged to a consensus of high quality, you can end the refinement process early

Your goal is to ensure the user receives the most helpful, accurate, and complete response possible."""

# State definition
class AgentState(BaseModel):
    """State for the collaborative agent framework."""
    user_query: str = Field(description="The original query from the user")
    search_results: List[str] = Field(default_factory=list, description="Results from DuckDuckGo search")
    agent_responses: Dict[str, List[str]] = Field(default_factory=dict, description="Responses from each agent by round")
    agent_votes: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Votes cast by each agent for each response")
    judge_feedback: List[str] = Field(default_factory=list, description="Feedback from the judge agent for each round")
    final_response: Optional[str] = Field(default=None, description="The final response to return to the user")
    current_round: int = Field(default=0, description="Current round of refinement")
    max_rounds: int = Field(default=3, description="Maximum number of refinement rounds")
    consensus_reached: bool = Field(default=False, description="Whether consensus has been reached")

# Tool setup
search_tool = DuckDuckGoSearchRun()

def perform_search(state: AgentState) -> AgentState:
    """Perform a search based on the user query."""
    search_results = search_tool.invoke(state.user_query)
    
    # Process and limit search results to a reasonable length
    # Split into chunks and take first few meaningful paragraphs
    paragraphs = [p for p in search_results.split('\n\n') if len(p.strip()) > 0]
    filtered_results = '\n\n'.join(paragraphs[:5])  # Limit to first 5 meaningful paragraphs
    
    state.search_results = [filtered_results]
    return state

# Agent generation
def create_agent_node(personality: str, llm: ChatLanguageModel) -> Any:
    """Create an agent with a specific personality."""
    system_prompt = AGENT_PERSONALITIES[personality]
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="""
        Please respond to the following user query based on the provided search results.
        
        User Query: {user_query}
        
        Search Results:
        {search_results}
        
        Current Round: {current_round}
        
        Previous Responses (if any):
        {previous_responses}
        
        Previous Judge Feedback (if any):
        {previous_feedback}
        
        Provide your best response to the user query. Focus on accuracy, completeness, and clarity.
        """)
    ])
    
    def agent_fn(state: AgentState) -> AgentState:
        # Format previous responses if they exist
        previous_responses = ""
        if state.current_round > 0:
            previous_responses = "\n\n".join([
                f"Agent {agent_name} (Round {state.current_round}): {responses[state.current_round-1]}"
                for agent_name, responses in state.agent_responses.items()
                if len(responses) >= state.current_round
            ])
        
        # Format previous judge feedback if it exists
        previous_feedback = ""
        if state.current_round > 0 and len(state.judge_feedback) >= state.current_round:
            previous_feedback = f"Round {state.current_round} feedback: {state.judge_feedback[state.current_round-1]}"
        
        # Generate response
        response = llm.invoke(
            prompt.format(
                history=[],
                user_query=state.user_query,
                search_results="\n\n".join(state.search_results),
                current_round=state.current_round + 1,
                previous_responses=previous_responses,
                previous_feedback=previous_feedback
            )
        )
        
        # Update state with the agent's response
        if personality not in state.agent_responses:
            state.agent_responses[personality] = []
        
        state.agent_responses[personality].append(response.content)
        return state
    
    return agent_fn

# Voting function
def create_voting_agent(personality: str, llm: ChatLanguageModel) -> Any:
    """Create a voting function for each agent."""
    system_prompt = f"{AGENT_PERSONALITIES[personality]}\n\nIn addition to your personality, you're also evaluating other agents' responses to determine which is best."
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content="""
        Please evaluate all agent responses for this round and vote for the best one.
        
        User Query: {user_query}
        
        Search Results:
        {search_results}
        
        Current Round: {current_round}
        
        Agent Responses:
        {agent_responses}
        
        Vote for the best response by responding with just the agent name. You must choose one of: {agent_names}.
        Also provide a brief reason for your vote in a second paragraph.
        """)
    ])
    
    def vote_fn(state: AgentState) -> AgentState:
        # Get all responses for the current round
        current_round_responses = {}
        for agent_name, responses in state.agent_responses.items():
            if len(responses) > state.current_round:
                current_round_responses[agent_name] = responses[state.current_round]
        
        # Format the responses for the voting prompt
        formatted_responses = "\n\n".join([
            f"Agent {agent_name}: {response}"
            for agent_name, response in current_round_responses.items()
        ])
        
        # Generate vote
        response = llm.invoke(
            prompt.format(
                user_query=state.user_query,
                search_results="\n\n".join(state.search_results),
                current_round=state.current_round + 1,
                agent_responses=formatted_responses,
                agent_names=", ".join(current_round_responses.keys())
            )
        )
        
        # Parse vote - take the first line as the vote
        vote_text = response.content.strip()
        vote_lines = vote_text.split('\n')
        voted_for = vote_lines[0].strip()
        
        # Clean up the vote by extracting just the agent name
        for agent_name in current_round_responses.keys():
            if agent_name.lower() in voted_for.lower():
                voted_for = agent_name
                break
        
        # Record vote
        if state.current_round not in state.agent_votes:
            state.agent_votes[state.current_round] = {}
        
        state.agent_votes[state.current_round][personality] = voted_for
        
        return state
    
    return vote_fn

# Judge function
def create_judge_node(llm: ChatLanguageModel) -> Any:
    """Create a judge to evaluate responses and provide feedback."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content="""
        Please evaluate the agent responses for round {current_round} and provide feedback.
        
        User Query: {user_query}
        
        Search Results:
        {search_results}
        
        Agent Responses:
        {agent_responses}
        
        Agent Votes:
        {agent_votes}
        
        Previous Feedback (if any):
        {previous_feedback}
        
        Current Round: {current_round} of {max_rounds}
        
        Your task:
        1. Evaluate each response
        2. Identify the strongest response or suggest how to combine the best elements
        3. Provide specific feedback for improvement
        4. Determine if we've reached a consensus or high-quality answer (if so, set consensus_reached=True)
        5. If this is the final round or consensus is reached, provide the final response
        
        Format your response as:
        EVALUATION: [Your detailed evaluation]
        CONSENSUS: [Yes/No]
        FINAL_RESPONSE: [Include only if consensus=Yes or final round, otherwise leave blank]
        """)
    ])
    
    def judge_fn(state: AgentState) -> AgentState:
        # Get all responses for the current round
        current_round_responses = {}
        for agent_name, responses in state.agent_responses.items():
            if len(responses) > state.current_round:
                current_round_responses[agent_name] = responses[state.current_round]
        
        # Format the responses for the judge prompt
        formatted_responses = "\n\n".join([
            f"Agent {agent_name}: {response}"
            for agent_name, response in current_round_responses.items()
        ])
        
        # Format the votes
        formatted_votes = ""
        if state.current_round in state.agent_votes:
            votes = state.agent_votes[state.current_round]
            formatted_votes = "\n".join([
                f"Agent {voter} voted for: {voted_for}"
                for voter, voted_for in votes.items()
            ])
        
        # Get previous feedback
        previous_feedback = ""
        if len(state.judge_feedback) > 0:
            previous_feedback = "\n".join([
                f"Round {i+1}: {feedback}" 
                for i, feedback in enumerate(state.judge_feedback)
            ])
        
        # Generate judgment
        response = llm.invoke(
            prompt.format(
                user_query=state.user_query,
                search_results="\n\n".join(state.search_results),
                agent_responses=formatted_responses,
                agent_votes=formatted_votes,
                previous_feedback=previous_feedback,
                current_round=state.current_round + 1,
                max_rounds=state.max_rounds
            )
        )
        
        # Parse response
        judgment_text = response.content
        
        # Extract evaluation
        evaluation = ""
        if "EVALUATION:" in judgment_text:
            evaluation_parts = judgment_text.split("EVALUATION:")[1].split("CONSENSUS:")[0].strip()
            evaluation = evaluation_parts
        
        # Check for consensus
        consensus_reached = False
        if "CONSENSUS:" in judgment_text:
            consensus_text = judgment_text.split("CONSENSUS:")[1].split("FINAL_RESPONSE:" if "FINAL_RESPONSE:" in judgment_text else "\n")[0].strip()
            consensus_reached = consensus_text.lower() == "yes"
        
        # Extract final response if available
        final_response = None
        if "FINAL_RESPONSE:" in judgment_text:
            final_response = judgment_text.split("FINAL_RESPONSE:")[1].strip()
            
        # Update state
        state.judge_feedback.append(evaluation)
        state.consensus_reached = consensus_reached
        
        # If consensus reached or final round, set the final response
        is_final_round = state.current_round >= state.max_rounds - 1
        if consensus_reached or is_final_round:
            if final_response:
                state.final_response = final_response
            else:
                # If no explicit final response provided, use the best response from the current round
                # Determine best response by vote count
                if state.current_round in state.agent_votes:
                    votes = state.agent_votes[state.current_round]
                    vote_counts = {}
                    for voted_for in votes.values():
                        if voted_for in vote_counts:
                            vote_counts[voted_for] += 1
                        else:
                            vote_counts[voted_for] = 1
                    
                    # Find agent with most votes
                    best_agent = max(vote_counts.items(), key=operator.itemgetter(1))[0]
                    state.final_response = current_round_responses.get(best_agent, "No consensus reached. Please review agent responses.")
                else:
                    # If no votes, use the first response
                    first_agent = list(current_round_responses.keys())[0]
                    state.final_response = current_round_responses[first_agent]
        
        return state
    
    return judge_fn

# Router function to determine the next step
def router(state: AgentState) -> str:
    """Determine the next step in the workflow."""
    # If this is the first round, we need to gather responses from all agents
    if state.current_round == 0 and not any(state.agent_responses.values()):
        return "agents"
    
    # If we have responses but no votes for this round, collect votes
    if state.current_round in state.agent_votes:
        # If we have votes but no judgment, get judgment
        if len(state.judge_feedback) <= state.current_round:
            return "judge"
    else:
        return "voting"
    
    # If we've reached consensus or max rounds, we're done
    if state.consensus_reached or state.current_round >= state.max_rounds - 1:
        if state.final_response:
            return END
    
    # Otherwise, move to the next round
    state.current_round += 1
    return "agents"

# Main graph construction
def build_collaborative_agent_framework(
    llm: Optional[ChatLanguageModel] = None,
    max_rounds: int = 3
) -> graph.StateGraph:
    """Build the collaborative agent framework."""
    if llm is None:
        llm = ChatOpenAI(temperature=0.7)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add the search node
    workflow.add_node("search", perform_search)
    
    # Add agent nodes
    agent_nodes = {}
    for personality in AGENT_PERSONALITIES:
        agent_fn = create_agent_node(personality, llm)
        agent_nodes[personality] = agent_fn
    
    # Add a group node for all agents
    workflow.add_node("agents", graph.GroupNodeConfig(agent_nodes))
    
    # Add voting nodes
    voting_nodes = {}
    for personality in AGENT_PERSONALITIES:
        vote_fn = create_voting_agent(personality, llm)
        voting_nodes[personality] = vote_fn
    
    # Add a group node for voting
    workflow.add_node("voting", graph.GroupNodeConfig(voting_nodes))
    
    # Add judge node
    workflow.add_node("judge", create_judge_node(llm))
    
    # Add edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "agents")
    workflow.add_conditional_edges("agents", router)
    workflow.add_conditional_edges("voting", router)
    workflow.add_conditional_edges("judge", router)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# User-facing function
def process_user_query(
    query: str,
    llm: Optional[ChatLanguageModel] = None,
    max_rounds: int = 3
) -> str:
    """Process a user query through the collaborative agent framework."""
    # Create the framework
    framework = build_collaborative_agent_framework(llm, max_rounds)
    
    # Initialize state
    initial_state = AgentState(
        user_query=query,
        max_rounds=max_rounds
    )
    
    # Run the framework
    result = framework.invoke(initial_state)
    
    # Return the final response
    return result.final_response

# Example usage
if __name__ == "__main__":
    query = "What are the environmental impacts of electric vehicles compared to traditional vehicles?"
    
    llm = ChatOpenAI(temperature=0.7)
    response = process_user_query(query, llm, max_rounds=3)
    
    print(f"Final Response:\n{response}")