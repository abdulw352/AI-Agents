import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go

# LangChain imports
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers.json import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import create_extraction_chain
from langchain.agents import AgentExecutor, create_json_chat_agent
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Set page config
st.set_page_config(
    page_title="AI Fitness Trainer",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define constants
DATA_FOLDER = os.path.join(os.getcwd(), "data")
CONFIG_FILE = os.path.join(DATA_FOLDER, "config.json")
os.makedirs(DATA_FOLDER, exist_ok=True)

# Define custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0fff4;
        border: 1px solid #9ae6b4;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fffaf0;
        border: 1px solid #fbd38d;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .metrics-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .trend-up {
        color: green;
    }
    .trend-down {
        color: red;
    }
    .trend-neutral {
        color: gray;
    }
    </style>
""", unsafe_allow_html=True)

# Define schema for our agents
class UserProfile(BaseModel):
    age: int = Field(description="User's age in years")
    weight: float = Field(description="User's weight in kg")
    height: float = Field(description="User's height in cm")
    sex: str = Field(description="User's biological sex (Male, Female, Other)")
    activity_level: str = Field(description="User's typical activity level")
    dietary_preferences: str = Field(description="User's dietary preferences")
    fitness_goals: str = Field(description="User's fitness goals")
    health_conditions: List[str] = Field(description="User's health conditions or limitations", default_factory=list)

class FitnessAnalysis(BaseModel):
    bmi: float = Field(description="Body Mass Index")
    bmi_category: str = Field(description="BMI category (Underweight, Normal, Overweight, Obese)")
    calorie_needs: Dict[str, float] = Field(description="Daily calorie needs for different goals")
    key_insights: List[str] = Field(description="Key insights from the data analysis")
    areas_of_improvement: List[str] = Field(description="Areas where user can improve")
    strengths: List[str] = Field(description="User's current strengths")

class FitnessRecommendation(BaseModel):
    summary: str = Field(description="Summary of fitness recommendations")
    workout_plan: Dict[str, Any] = Field(description="Detailed workout plan")
    dietary_advice: Dict[str, Any] = Field(description="Dietary recommendations")
    habit_changes: List[str] = Field(description="Habit changes to improve fitness")
    progress_metrics: List[str] = Field(description="Metrics to track progress")

# Initialize our LLM
@st.cache_resource
def get_llm():
    return Ollama(model="llama3.1", temperature=0.1)

# Helper functions
def calculate_bmi(weight: float, height: float) -> float:
    """Calculate BMI from weight (kg) and height (cm)"""
    height_m = height / 100
    return weight / (height_m * height_m)

def get_bmi_category(bmi: float) -> str:
    """Get BMI category"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def calculate_calorie_needs(profile: UserProfile) -> Dict[str, float]:
    """Calculate daily calorie needs based on profile"""
    # Base calculation using Mifflin-St Jeor Equation
    if profile.sex.lower() == "male":
        bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age + 5
    else:
        bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age - 161
    
    # Activity multipliers
    activity_multipliers = {
        "sedentary": 1.2,
        "lightly active": 1.375,
        "moderately active": 1.55,
        "very active": 1.725,
        "extremely active": 1.9
    }
    
    # Get the appropriate multiplier
    multiplier = activity_multipliers.get(profile.activity_level.lower(), 1.5)
    
    # Calculate maintenance calories
    maintenance = bmr * multiplier
    
    return {
        "maintenance": round(maintenance),
        "weight_loss": round(maintenance - 500),
        "weight_gain": round(maintenance + 500)
    }

def load_user_data(file_path: str) -> pd.DataFrame:
    """Load user fitness data from a file"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".csv":
        return pd.read_csv(file_path)
    elif file_extension in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return pd.DataFrame()

def save_user_profile(profile: Dict[str, Any]):
    """Save user profile to config file"""
    config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    
    config["user_profile"] = profile
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def load_user_profile() -> Dict[str, Any]:
    """Load user profile from config file"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get("user_profile", {})
    return {}

# Define agent components
class FitnessAgentState(dict):
    """State for our fitness agent graph"""
    user_profile: Dict[str, Any]
    fitness_data: pd.DataFrame
    analysis: Dict[str, Any] = None
    recommendation: Dict[str, Any] = None
    question: str = None
    answer: str = None

# Agent nodes (functions)
def data_analysis_node(state: FitnessAgentState) -> FitnessAgentState:
    """Analyze user fitness data"""
    llm = get_llm()
    
    # Extract basic statistics from data
    data_stats = {}
    if not state["fitness_data"].empty:
        # Calculate some basic stats
        for column in state["fitness_data"].select_dtypes(include=[np.number]).columns:
            data_stats[column] = {
                "mean": state["fitness_data"][column].mean(),
                "min": state["fitness_data"][column].min(),
                "max": state["fitness_data"][column].max(),
                "trend": "up" if state["fitness_data"][column].iloc[-1] > state["fitness_data"][column].iloc[0] else "down"
            }
    
    # Calculate BMI and other metrics
    profile = state["user_profile"]
    bmi = calculate_bmi(profile.get("weight", 70), profile.get("height", 170))
    bmi_category = get_bmi_category(bmi)
    calorie_needs = calculate_calorie_needs(UserProfile(**profile))
    
    # Prepare context for LLM
    context = f"""
    User Profile: {json.dumps(profile, indent=2)}
    Data Statistics: {json.dumps(data_stats, indent=2)}
    BMI: {bmi:.2f}
    BMI Category: {bmi_category}
    Daily Calorie Needs: {json.dumps(calorie_needs, indent=2)}
    """
    
    # Create prompt for analysis
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a fitness analysis expert. Analyze the user's fitness data and profile to identify key insights, areas of improvement, and strengths."),
        ("human", "Here is the user data:\n{context}")
    ])
    
    # Create extraction chain
    parser = JsonOutputParser(pydantic_object=FitnessAnalysis)
    analysis_chain = analysis_prompt | llm | parser
    
    # Run the chain
    analysis = analysis_chain.invoke({"context": context})
    
    # Update state
    state["analysis"] = analysis.dict()
    return state

def recommendation_node(state: FitnessAgentState) -> FitnessAgentState:
    """Generate fitness recommendations based on analysis"""
    llm = get_llm()
    
    # Prepare context
    context = f"""
    User Profile: {json.dumps(state['user_profile'], indent=2)}
    Analysis: {json.dumps(state['analysis'], indent=2)}
    """
    
    # Create prompt for recommendations
    recommendation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a personal fitness trainer. Based on the user's profile and analysis, provide detailed recommendations for improving their fitness."),
        ("human", "Here is the user data and analysis:\n{context}")
    ])
    
    # Create extraction chain
    parser = JsonOutputParser(pydantic_object=FitnessRecommendation)
    recommendation_chain = recommendation_prompt | llm | parser
    
    # Run the chain
    recommendation = recommendation_chain.invoke({"context": context})
    
    # Update state
    state["recommendation"] = recommendation.dict()
    return state

def qa_node(state: FitnessAgentState) -> FitnessAgentState:
    """Answer user questions about fitness and health"""
    llm = get_llm()
    
    # Prepare context
    context = f"""
    User Profile: {json.dumps(state['user_profile'], indent=2)}
    Analysis: {json.dumps(state['analysis'], indent=2)}
    Recommendation: {json.dumps(state['recommendation'], indent=2)}
    Question: {state['question']}
    """
    
    # Create prompt for QA
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a fitness expert assistant. Answer the user's question based on their profile, analysis, and recommendations."),
        ("human", "Context:\n{context}\n\nPlease answer my question: {question}")
    ])
    
    # Create chain
    qa_chain = qa_prompt | llm | StrOutputParser()
    
    # Run the chain
    answer = qa_chain.invoke({"context": context, "question": state["question"]})
    
    # Update state
    state["answer"] = answer
    return state

# Create the agent graph
def create_fitness_agent_graph():
    """Create the fitness agent graph using LangGraph"""
    workflow = StateGraph(FitnessAgentState)
    
    # Add nodes
    workflow.add_node("analyze_data", data_analysis_node)
    workflow.add_node("generate_recommendations", recommendation_node)
    workflow.add_node("answer_question", qa_node)
    
    # Add edges
    workflow.add_edge("analyze_data", "generate_recommendations")
    workflow.add_edge("generate_recommendations", END)
    workflow.add_edge("answer_question", END)
    
    # Conditional edges
    workflow.add_conditional_edges(
        "start",
        lambda state: "answer_question" if state["question"] else "analyze_data"
    )
    
    return workflow.compile()

# Create the agent executor
@st.cache_resource
def get_agent_executor():
    """Get the agent executor with graph"""
    return create_fitness_agent_graph()

# Streamlit UI Components
def render_profile_form():
    """Render the user profile form"""
    st.header("👤 Your Profile")
    
    # Load existing profile
    profile = load_user_profile()
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, step=1, value=profile.get("age", 30))
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, step=0.1, value=profile.get("height", 170.0))
        activity_level = st.selectbox(
            "Activity Level",
            options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
            index=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"].index(profile.get("activity_level", "Moderately Active"))
        )
        dietary_preferences = st.selectbox(
            "Dietary Preferences",
            options=["No Restrictions", "Vegetarian", "Vegan", "Keto", "Gluten Free", "Low Carb", "Dairy Free"],
            index=["No Restrictions", "Vegetarian", "Vegan", "Keto", "Gluten Free", "Low Carb", "Dairy Free"].index(profile.get("dietary_preferences", "No Restrictions"))
        )

    with col2:
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, step=0.1, value=profile.get("weight", 70.0))
        sex = st.selectbox(
            "Sex", 
            options=["Male", "Female", "Other"],
            index=["Male", "Female", "Other"].index(profile.get("sex", "Male"))
        )
        fitness_goals = st.selectbox(
            "Fitness Goals",
            options=["Lose Weight", "Gain Muscle", "Improve Endurance", "Maintain Fitness", "Strength Training"],
            index=["Lose Weight", "Gain Muscle", "Improve Endurance", "Maintain Fitness", "Strength Training"].index(profile.get("fitness_goals", "Maintain Fitness"))
        )
        health_conditions = st.multiselect(
            "Health Conditions (if any)",
            options=["None", "Diabetes", "Hypertension", "Heart Disease", "Joint Pain", "Back Pain", "Asthma", "Other"],
            default=profile.get("health_conditions", ["None"])
        )
    
    if st.button("Save Profile", use_container_width=True):
        updated_profile = {
            "age": age,
            "weight": weight,
            "height": height,
            "sex": sex,
            "activity_level": activity_level,
            "dietary_preferences": dietary_preferences,
            "fitness_goals": fitness_goals,
            "health_conditions": health_conditions if "None" not in health_conditions else []
        }
        
        save_user_profile(updated_profile)
        st.success("✅ Profile saved successfully!")
        return updated_profile
    
    return profile