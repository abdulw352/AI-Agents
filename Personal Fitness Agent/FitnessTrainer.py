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

def render_data_upload():
    """Render the data upload section"""
    st.header("📊 Your Fitness Data")
    
    st.info("Upload your fitness data CSV or Excel file to get personalized insights. The file should contain workout logs, weight tracking, or other fitness metrics.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        # Save the file to data folder
        file_name = uploaded_file.name
        file_path = os.path.join(DATA_FOLDER, file_name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ File saved to {file_path}")
        
        # Load and return the data
        data = load_user_data(file_path)
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        return data
    
    # Check if data files exist in the data folder
    existing_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(('.csv', '.xlsx', '.xls'))]
    
    if existing_files:
        st.subheader("Previous Data Files")
        selected_file = st.selectbox("Select a file to use", options=existing_files)
        
        if selected_file:
            file_path = os.path.join(DATA_FOLDER, selected_file)
            data = load_user_data(file_path)
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            return data
    
    # Create sample data for demonstration
    if st.checkbox("Use sample data for demonstration"):
        # Generate sample fitness data
        date_range = pd.date_range(end=datetime.now(), periods=30).tolist()
        
        sample_data = pd.DataFrame({
            'Date': date_range,
            'Weight (kg)': np.linspace(80, 77, 30) + np.random.normal(0, 0.5, 30),
            'Steps': np.random.randint(5000, 15000, 30),
            'Calories Burned': np.random.randint(1800, 3000, 30),
            'Workout Duration (min)': np.random.randint(0, 120, 30),
            'Sleep (hours)': np.random.uniform(5, 9, 30).round(1)
        })
        
        # Save sample data
        sample_path = os.path.join(DATA_FOLDER, "sample_fitness_data.csv")
        sample_data.to_csv(sample_path, index=False)
        
        st.success(f"✅ Sample data created at {sample_path}")
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(sample_data.head())
        
        return sample_data
    
    return pd.DataFrame()  # Return empty dataframe if no data is uploaded

def visualize_data(data: pd.DataFrame):
    """Create visualizations for the fitness data"""
    if data.empty:
        return
    
    st.header("📈 Data Visualization")
    
    # Determine the date column
    date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
    date_col = date_cols[0] if date_cols else None
    
    if date_col:
        # Ensure date column is datetime
        data[date_col] = pd.to_datetime(data[date_col])
        
        # Sort by date
        data = data.sort_values(by=date_col)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Trends", "Correlations", "Summary"])
    
    with tab1:
        st.subheader("Fitness Metrics Over Time")
        
        if date_col:
            # Get numeric columns for plotting
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_metric = st.selectbox(
                    "Select metric to visualize",
                    options=numeric_cols
                )
                
                # Plot the selected metric over time
                fig = px.line(
                    data, 
                    x=date_col, 
                    y=selected_metric,
                    title=f"{selected_metric} Over Time",
                    markers=True
                )
                
                # Add trend line
                fig.add_trace(
                    go.Scatter(
                        x=data[date_col],
                        y=data[selected_metric].rolling(window=7).mean(),
                        mode='lines',
                        name='7-day Moving Average',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add progress metrics
                if len(data) > 1:
                    first_value = data[selected_metric].iloc[0]
                    last_value = data[selected_metric].iloc[-1]
                    change = last_value - first_value
                    percent_change = (change / first_value) * 100 if first_value != 0 else 0
                    
                    # Determine if increase is good or bad based on the metric name
                    positive_metrics = ["steps", "workout", "duration", "sleep", "water"]
                    negative_metrics = ["weight", "fat", "bmi", "waist"]
                    
                    is_positive_metric = any(term in selected_metric.lower() for term in positive_metrics)
                    is_negative_metric = any(term in selected_metric.lower() for term in negative_metrics)
                    
                    if (is_positive_metric and change > 0) or (is_negative_metric and change < 0):
                        trend_class = "trend-up"
                        trend_icon = "📈"
                    elif (is_positive_metric and change < 0) or (is_negative_metric and change > 0):
                        trend_class = "trend-down"
                        trend_icon = "📉"
                    else:
                        trend_class = "trend-neutral"
                        trend_icon = "➡️"
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label=f"Starting {selected_metric}",
                            value=f"{first_value:.1f}"
                        )
                    
                    with col2:
                        st.metric(
                            label=f"Current {selected_metric}",
                            value=f"{last_value:.1f}",
                            delta=f"{change:.1f}"
                        )
                    
                    with col3:
                        st.markdown(
                            f"""
                            <div class="metrics-card">
                                <h4>Overall Trend</h4>
                                <p><span class="{trend_class}">{trend_icon} {percent_change:.1f}%</span></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
    
    with tab2:
        st.subheader("Correlations Between Metrics")
        
        # Get numeric columns for correlation
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_metric = st.selectbox("X-axis metric", options=numeric_cols, key="x_metric")
            
            with col2:
                y_metric = st.selectbox("Y-axis metric", options=numeric_cols, key="y_metric", index=1 if len(numeric_cols) > 1 else 0)
            
            # Create scatter plot
            fig = px.scatter(
                data,
                x=x_metric,
                y=y_metric,
                trendline="ols",
                title=f"Correlation between {x_metric} and {y_metric}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation
            corr = data[x_metric].corr(data[y_metric])
            
            st.info(f"Correlation coefficient: {corr:.2f}")
            
            # Interpretation
            if abs(corr) > 0.7:
                st.success(f"Strong {'positive' if corr > 0 else 'negative'} correlation")
            elif abs(corr) > 0.3:
                st.info(f"Moderate {'positive' if corr > 0 else 'negative'} correlation")
            else:
                st.warning("Weak correlation")
        
        # Show full correlation matrix
        if len(numeric_cols) > 2:
            with st.expander("View full correlation matrix"):
                corr_matrix = data[numeric_cols].corr()
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                plt.tight_layout()
                
                st.pyplot(fig)
    
    with tab3:
        st.subheader("Summary Statistics")
        
        # Get numeric columns for summary
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            summary_df = data[numeric_cols].describe().T
            
            # Round to 2 decimal places
            summary_df = summary_df.round(2)
            
            st.dataframe(summary_df)
            
            # Show weekly averages if date column exists
            if date_col:
                with st.expander("View weekly averages"):
                    # Convert date to week start
                    data['Week'] = pd.to_datetime(data[date_col]).dt.to_period('W').dt.start_time
                    
                    # Group by week and calculate mean
                    weekly_avg = data.groupby('Week')[numeric_cols].mean().reset_index()
                    
                    # Round to 2 decimal places
                    weekly_avg[numeric_cols] = weekly_avg[numeric_cols].round(2)
                    
                    st.dataframe(weekly_avg)

def display_analysis_results(profile: Dict[str, Any], analysis: Dict[str, Any]):
    """Display the fitness analysis results"""
    if not analysis:
        return
    
    st.header("🔍 Your Fitness Analysis")
    
    # Display BMI and category
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("BMI", f"{analysis.get('bmi', 0):.1f}")
    
    with col2:
        st.metric("BMI Category", analysis.get("bmi_category", "Unknown"))
    
    with col3:
        # Determine which calorie number to show based on goals
        goals = profile.get("fitness_goals", "").lower()
        
        if "lose" in goals or "weight loss" in goals:
            calorie_key = "weight_loss"
            calorie_label = "Weight Loss Target"
        elif "gain" in goals or "muscle" in goals:
            calorie_key = "weight_gain"
            calorie_label = "Weight Gain Target"
        else:
            calorie_key = "maintenance"
            calorie_label = "Maintenance Calories"
        
        daily_calories = analysis.get("calorie_needs", {}).get(calorie_key, 2000)
        st.metric(calorie_label, f"{daily_calories} kcal/day")
    
    # Display key insights
    st.subheader("Key Insights")
    
    for i, insight in enumerate(analysis.get("key_insights", [])):
        st.info(insight)
    
    # Areas of improvement and strengths
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Areas to Improve")
        for i, area in enumerate(analysis.get("areas_of_improvement", [])):
            st.warning(area)
    
    with col2:
        st.subheader("Your Strengths")
        for i, strength in enumerate(analysis.get("strengths", [])):
            st.success(strength)

def display_recommendations(recommendation: Dict[str, Any]):
    """Display fitness recommendations"""
    if not recommendation:
        return
    
    st.header("💪 Your Personalized Fitness Plan")
    
    # Display summary
    st.info(recommendation.get("summary", ""))
    
    # Display workout plan
    with st.expander("🏋️‍♂️ Workout Plan", expanded=True):
        workout_plan = recommendation.get("workout_plan", {})
        
        if workout_plan:
            # Check if workout plan has a day-by-day structure
            if any(key.lower().startswith(("day", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")) for key in workout_plan.keys()):
                # Display as a weekly plan
                for day, exercises in workout_plan.items():
                    st.subheader(day)
                    
                    if isinstance(exercises, str):
                        st.write(exercises)
                    elif isinstance(exercises, list):
                        for exercise in exercises:
                            st.write(f"- {exercise}")
                    elif isinstance(exercises, dict):
                        for exercise, details in exercises.items():
                            st.write(f"**{exercise}:** {details}")
            else:
                # Display as a general plan
                for section, content in workout_plan.items():
                    st.subheader(section)
                    
                    if isinstance(content, str):
                        st.write(content)
                    elif isinstance(content, list):
                        for item in content:
                            st.write(f"- {item}")
                    elif isinstance(content, dict):
                        for key, value in content.items():
                            st.write(f"**{key}:** {value}")
    
    # Display dietary advice
    with st.expander("🍽️ Dietary Recommendations", expanded=True):
        dietary_advice = recommendation.get("dietary_advice", {})
        
        if dietary_advice:
            for section, content in dietary_advice.items():
                st.subheader(section)
                
                if isinstance(content, str):
                    st.write(content)
                elif isinstance(content, list):
                    for item in content:
                        st.write(f"- {item}")
                elif isinstance(content, dict):
                    for key, value in content.items():
                        st.write(f"**{key}:** {value}")
    
    # Display habit changes
    with st.expander("🔄 Recommended Habit Changes"):
        habit_changes = recommendation.get("habit_changes", [])
        
        if habit_changes:
            for habit in habit_changes:
                st.success(habit)
    
    # Display progress metrics to track
    with st.expander("📊 Progress Metrics to Track"):
        metrics = recommendation.get("progress_metrics", [])
        
        if metrics:
            for metric in metrics:
                st.info(metric)

def display_qa_section(profile: Dict[str, Any], analysis: Dict[str, Any], recommendation: Dict[str, Any]):
    """Display Q&A section for user questions"""
    st.header("❓ Ask Your Fitness Coach")
    
    # Initialize conversation history if not exists
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    
    # Display conversation history
    for q, a in st.session_state.qa_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Coach:** {a}")
    
    # Input for new questions
    question = st.text_input("What would you like to know about your fitness plan?")
    
    if st.button("Ask", use_container_width=True) and question:
        # Check if we have analysis and recommendations
        if not analysis or not recommendation:
            st.warning("⚠️ Please generate your fitness analysis and recommendations first.")
            return
        
        with st.spinner("Thinking..."):
            # Run QA node through agent
            agent_executor = get_agent_executor()
            state = FitnessAgentState(
                user_profile=profile,
                fitness_data=pd.DataFrame(),  # Not used in QA
                analysis=analysis,
                recommendation=recommendation,
                question=question
            )
            result = agent_executor.invoke(state)
            
            # Get the answer
            answer = result["answer"]
            
            # Add to history
            st.session_state.qa_history.append((question, answer))
            
            # Display the answer
            st.markdown(f"**You:** {question}")
            st.markdown(f"**Coach:** {answer}")

def export_data(profile: Dict[str, Any], analysis: Dict[str, Any], recommendation: Dict[str, Any]):
    """Export fitness plan to a file"""
    if not analysis or not recommendation:
        return
    
    st.header("📋 Export Your Fitness Plan")
    
    export_format = st.selectbox("Export Format", ["PDF", "JSON", "Markdown"])
    
    if st.button("Generate Export", use_container_width=True):
        # Create export folder if not exists
        export_folder = os.path.join(DATA_FOLDER, "exports")
        os.makedirs(export_folder, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == "JSON":
            # Combine all data
            export_data = {
                "user_profile": profile,
                "fitness_analysis": analysis,
                "fitness_recommendation": recommendation,
                "generated_at": timestamp
            }
            
            # Save to file
            file_path = os.path.join(export_folder, f"fitness_plan_{timestamp}.json")
            
            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=4)
            
            st.success(f"✅ Fitness plan exported to {file_path}")
        
        elif export_format == "Markdown":
            # Create markdown content
            md_content = f"""# Your Personal Fitness Plan
Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}

## User Profile
- Age: {profile.get('age')}
- Weight: {profile.get('weight')} kg
- Height: {profile.get('height')} cm
- Sex: {profile.get('sex')}
- Activity Level: {profile.get('activity_level')}
- Dietary Preferences: {profile.get('dietary_preferences')}
- Fitness Goals: {profile.get('fitness_goals')}

## Fitness Analysis
- BMI: {analysis.get('bmi'):.1f}
- BMI Category: {analysis.get('bmi_category')}

### Key Insights
{chr(10).join(f"- {insight}" for insight in analysis.get('key_insights', []))}

### Areas to Improve
{chr(10).join(f"- {area}" for area in analysis.get('areas_of_improvement', []))}

### Strengths
{chr(10).join(f"- {strength}" for strength in analysis.get('strengths', []))}

## Workout Plan
{recommendation.get('summary')}

"""
            # Add workout plan details
            workout_plan = recommendation.get('workout_plan', {})
            if workout_plan:
                md_content += "\n### Workout Details\n"
                for key, value in workout_plan.items():
                    md_content += f"\n#### {key}\n"
                    if isinstance(value, str):
                        md_content += f"{value}\n"
                    elif isinstance(value, list):
                        md_content += "\n".join(f"- {item}" for item in value) + "\n"
                    elif isinstance(value, dict):
                        md_content += "\n".join(f"- **{k}**: {v}" for k, v in value.items()) + "\n"
            
            # Add dietary advice
            dietary_advice = recommendation.get('dietary_advice', {})
            if dietary_advice:
                md_content += "\n## Dietary Recommendations\n"
                for key, value in dietary_advice.items():
                    md_content += f"\n### {key}\n"
                    if isinstance(value, str):
                        md_content += f"{value}\n"
                    elif isinstance(value, list):
                        md_content += "\n".join(f"- {item}" for item in value) + "\n"
                    elif isinstance(value, dict):
                        md_content += "\n".join(f"- **{k}**: {v}" for k, v in value.items()) + "\n"
            
            # Add habit changes
            habit_changes = recommendation.get('habit_changes', [])
            if habit_changes:
                md_content += "\n## Recommended Habit Changes\n"
                md_content += "\n".join(f"- {habit}" for habit in habit_changes) + "\n"
            
            # Add progress metrics
            metrics = recommendation.get('progress_metrics', [])
            if metrics:
                md_content += "\n## Progress Metrics to Track\n"
                md_content += "\n".join(f"- {metric}" for metric in metrics) + "\n"
            
            # Save to file
            file_path = os.path.join(export_folder, f"fitness_plan_{timestamp}.md")
            
            with open(file_path, "w") as f:
                f.write(md_content)
            
            st.success(f"✅ Fitness plan exported to {file_path}")
        
        elif export_format == "PDF":
            # For PDF, we'll use a simple approach with markdown-to-pdf
            # In a production app, you'd use a more robust PDF generation library
            st.warning("PDF export would require additional libraries. For now, please use Markdown export.")

def main():
    """Main function for the Streamlit app"""
    # Initialize session state
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = None
    
    st.title("💪 AI Fitness Trainer")
    st.markdown("""
        <div style='background-color: #1E88E5; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem; color: white;'>
        Get personalized fitness analysis and recommendations tailored to your goals and unique profile.
        Our AI-powered system analyzes your data to create the perfect fitness plan just for you.
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for LLM configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check if Ollama is installed and running
        try:
            llm = get_llm()
            st.success("✅ Connected to Ollama with Llama3.1")
        except Exception as e:
            st.error(f"❌ Error connecting to Ollama: {str(e)}")
            st.info("""
                Please make sure Ollama is installed and running with Llama3.1 model.
                
                Installation instructions:
                1. Download Ollama from https://ollama.ai/
                2. Run `ollama pull llama3.1` to download the model
            """)
            return
        
        # Display app sections
        st.header("📑 Sections")
        st.markdown("""
            - User Profile
            - Data Upload
            - Data Visualization
            - Fitness Analysis
            - Recommendations
            - Ask Your Coach
            - Export Plan
        """)
        
        # Add reset button
        if st.button("Reset All Data", use_container_width=True):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.success("✅ All data reset!")
            st.rerun()
    
    # User profile section
    profile = render_profile_form()
    
    # Data upload section
    fitness_data = render_data_upload()
    
    # Data visualization if data is available
    if not fitness_data.empty:
        visualize_data(fitness_data)
    
    # Generate analysis button
    if profile and st.button("🔍 Generate Analysis & Recommendations", use_container_width=True):
        with st.spinner("Analyzing your fitness data and creating personalized recommendations..."):
            # Create initial state
            initial_state = FitnessAgentState(
                user_profile=profile,
                fitness_data=fitness_data,
                analysis=None,
                recommendation=None
            )
            
            # Run agent
            agent_executor = get_agent_executor()
            result = agent_executor.invoke(initial_state)
            
            # Store results in session state
            st.session_state.analysis_results = result["analysis"]
            st.session_state.recommendations = result["recommendation"]
            
            st.success("✅ Analysis and recommendations generated!")
    
    # Display analysis and recommendations if available
    if st.session_state.analysis_results and st.session_state.recommendations:
        display_analysis_results(profile, st.session_state.analysis_results)
        display_recommendations(st.session_state.recommendations)
        
        # Display QA section
        display_qa_section(profile, st.session_state.analysis_results, st.session_state.recommendations)
        
        # Export section
        export_data(profile, st.session_state.analysis_results, st.session_state.recommendations)

if __name__ == "__main__":
    main()