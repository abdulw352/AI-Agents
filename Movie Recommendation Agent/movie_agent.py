import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
import requests
from duckduckgo_search import DDGS
from io import StringIO
import re
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from langchain_community.llms import Ollama
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize Ollama LLM
llm = Ollama(
    model="llama3.1",
    temperature=0.7,
)

# Define pydantic models for structured output
class Movie(BaseModel):
    title: str = Field(description="Title of the movie")
    year: Optional[int] = Field(description="Year of release", default=None)
    genre: Optional[str] = Field(description="Genre of the movie", default=None)
    description: Optional[str] = Field(description="Brief plot description", default=None)
    why_recommended: str = Field(description="Reason why this movie is recommended")

class MovieRecommendations(BaseModel):
    recommendations: List[Movie] = Field(description="List of recommended movies")

# Initialize search tool
search_tool = DuckDuckGoSearchRun()

# Data Processing Functions
def load_data(file_path):
    """Load data from various formats (CSV, Excel)"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    else:
        st.error("Unsupported file format. Please upload CSV or Excel files.")
        return None

def preprocess_movie_data(df):
    """Clean and prepare the data for analysis"""
    # Handle common issues in movie datasets
    if 'title' in df.columns:
        # Extract year from title if needed
        if 'year' not in df.columns:
            df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype('float')
        
        # Clean title
        df['title'] = df['title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True)
    
    # Convert timestamps to datetime if present
    date_columns = df.columns[df.columns.str.contains('date', case=False)]
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass
            
    # Handle missing values
    df = df.fillna({
        'rating': df['rating'].mean() if 'rating' in df.columns else None,
        'genre': 'Unknown' if 'genre' in df.columns else None
    })
    
    return df

def analyze_data(df):
    """Generate insights from the user's movie watching data"""
    results = {}
    
    # Check if necessary columns exist
    if 'rating' in df.columns:
        results['avg_rating'] = df['rating'].mean()
        results['top_rated'] = df.nlargest(5, 'rating')['title'].tolist()
    
    if 'genre' in df.columns:
        results['genre_distribution'] = df['genre'].value_counts().to_dict()
        results['favorite_genre'] = df['genre'].value_counts().index[0]
    
    if 'year' in df.columns:
        results['year_distribution'] = df['year'].value_counts().to_dict()
        results['era_preference'] = df['year'].mean()
    
    return results

def visualize_data(df):
    """Create visualizations for the movie data"""
    figs = []
    
    if 'genre' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        genre_counts = df['genre'].value_counts().nlargest(10)
        sns.barplot(x=genre_counts.index, y=genre_counts.values, ax=ax)
        ax.set_title('Top 10 Movie Genres Watched')
        ax.set_xlabel('Genre')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        figs.append(fig)
    
    if 'year' in df.columns and df['year'].notna().any():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['year'].dropna(), bins=20, ax=ax)
        ax.set_title('Distribution of Movie Release Years')
        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        plt.tight_layout()
        figs.append(fig)
    
    if 'rating' in df.columns and df['rating'].notna().any():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['rating'].dropna(), bins=10, ax=ax)
        ax.set_title('Distribution of Ratings')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        plt.tight_layout()
        figs.append(fig)
        
        if 'genre' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            top_genres = df['genre'].value_counts().nlargest(5).index
            genre_data = df[df['genre'].isin(top_genres)]
            sns.boxplot(x='genre', y='rating', data=genre_data, ax=ax)
            ax.set_title('Ratings by Top 5 Genres')
            ax.set_xlabel('Genre')
            ax.set_ylabel('Rating')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            figs.append(fig)
    
    return figs

# Recommendation Engine Functions
def generate_content_based_recommendations(df, user_preferences, n=5):
    """Generate recommendations based on user's watching history"""
    if 'genre' not in df.columns or 'rating' not in df.columns:
        return []
    
    # Use LLM to generate recommendations based on user preferences
    favorite_genres = user_preferences['genre_distribution']
    top_genres = sorted(favorite_genres.items(), key=lambda x: x[1], reverse=True)[:3]
    top_genres_str = ", ".join([g[0] for g in top_genres])
    
    favorite_movies = ""
    if 'top_rated' in user_preferences:
        favorite_movies = ", ".join(user_preferences['top_rated'][:3])
    
    template = """
    Based on the user's movie preferences:
    - Favorite genres: {top_genres}
    - Highly rated movies: {favorite_movies}
    - Average rating given: {avg_rating}
    
    Recommend {n} movies that the user might enjoy but hasn't watched yet.
    For each movie, provide the title, release year, genre, a brief description, and why you're recommending it.
    """
    
    prompt = PromptTemplate(
        input_variables=["top_genres", "favorite_movies", "avg_rating", "n"],
        template=template
    )
    
    parser = PydanticOutputParser(pydantic_object=MovieRecommendations)
    
    recommendation_chain = (
        prompt
        | llm
        | parser
    )
    
    recommendations = recommendation_chain.invoke({
        "top_genres": top_genres_str,
        "favorite_movies": favorite_movies,
        "avg_rating": user_preferences.get('avg_rating', 'unknown'),
        "n": n
    })
    
    return recommendations.recommendations

# Chat-based recommendation system
class MovieRecommendationState(dict):
    """State for the movie recommendation graph"""
    def __init__(
        self,
        query: str = "",
        search_results: str = "",
        recommendations: Optional[List[Movie]] = None,
        response: str = "",
    ):
        self.query = query
        self.search_results = search_results
        self.recommendations = recommendations or []
        self.response = response
        
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

def web_search(state):
    """Search the web for movie information"""
    query = f"movie recommendations similar to {state.query}"
    search_results = search_tool.run(query)
    return {"search_results": search_results}

def generate_recommendations(state):
    """Generate movie recommendations based on the query and search results"""
    system_prompt = """
    You are a movie recommendation expert. Use the search results and your knowledge to recommend movies
    based on the user's query. For each movie, provide the title, release year, genre, a brief description, 
    and why you're recommending it based on their query.
    """
    
    human_prompt = """
    User Query: {query}
    
    Search Results: {search_results}
    
    Based on this information, recommend 3-5 movies that match the user's query.
    Format your response as a structured list of movies.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke({
        "query": state.query,
        "search_results": state.search_results
    })
    
    # Extract structured recommendations from the response
    # (In a production system, you might use a more robust parser)
    pattern = r"(?:^|\n)(?:\d+\.\s*|\*\s*)([^(\n]+)(?:\((\d{4})\))?(?:\s*-\s*([^:\n]+))?(?::\s*(.+?))?(?=\n(?:\d+\.\s*|\*\s*)|$)"
    matches = re.finditer(pattern, response, re.MULTILINE)
    
    recommendations = []
    for match in matches:
        title = match.group(1).strip() if match.group(1) else "Unknown Title"
        year = int(match.group(2)) if match.group(2) else None
        genre = match.group(3).strip() if match.group(3) else None
        description = match.group(4).strip() if match.group(4) else None
        
        movie = Movie(
            title=title,
            year=year,
            genre=genre,
            description=description,
            why_recommended=f"This matches your request for: {state.query}"
        )
        recommendations.append(movie)
    
    return {"recommendations": recommendations, "response": response}

def create_recommendation_graph():
    """Create the LangGraph for recommendation"""
    workflow = StateGraph(MovieRecommendationState)
    
    # Add nodes
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate_recommendations", generate_recommendations)
    
    # Add edges
    workflow.add_edge("web_search", "generate_recommendations")
    workflow.add_edge("generate_recommendations", END)
    
    # Set entry point
    workflow.set_entry_point("web_search")
    
    return workflow.compile()

# Streamlit UI
def create_ui():
    st.set_page_config(page_title="Movie Recommendation System", layout="wide")
    
    st.title("🎬 Movie Recommendation System")
    
    tab1, tab2 = st.tabs(["Data Analysis", "Chat Recommendations"])
    
    # Data Analysis Tab
    with tab1:
        st.header("Upload and Analyze Your Movie Data")
        
        uploaded_file = st.file_uploader("Upload your movie watching history (CSV or Excel)", 
                                        type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            # Load and process data
            df = load_data(uploaded_file)
            
            if df is not None:
                df = preprocess_movie_data(df)
                
                # Display sample data
                st.subheader("Preview of your data")
                st.dataframe(df.head())
                
                # Analyze data
                analysis_results = analyze_data(df)
                
                # Display insights
                st.subheader("Insights from your movie watching history")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'avg_rating' in analysis_results:
                        st.metric("Average Rating", f"{analysis_results['avg_rating']:.1f}/10")
                    
                    if 'favorite_genre' in analysis_results:
                        st.metric("Favorite Genre", analysis_results['favorite_genre'])
                
                with col2:
                    if 'era_preference' in analysis_results:
                        era = int(analysis_results['era_preference'])
                        st.metric("Era Preference", f"{era}s")
                    
                    if 'top_rated' in analysis_results:
                        st.write("Top Rated Movies:")
                        for movie in analysis_results['top_rated'][:3]:
                            st.write(f"- {movie}")
                
                # Visualizations
                st.subheader("Visualizations")
                visualizations = visualize_data(df)
                
                for fig in visualizations:
                    st.pyplot(fig)
                
                # Generate recommendations
                st.subheader("Recommended Movies Based on Your History")
                recommendations = generate_content_based_recommendations(df, analysis_results)
                
                for i, movie in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {movie.title} ({movie.year if movie.year else 'Unknown Year'})"):
                        st.write(f"**Genre:** {movie.genre if movie.genre else 'Unknown'}")
                        if movie.description:
                            st.write(f"**Description:** {movie.description}")
                        st.write(f"**Why we recommend it:** {movie.why_recommended}")
    
    # Chat Recommendations Tab
    with tab2:
        st.header("Chat for Movie Recommendations")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your movie recommendation assistant. Tell me what kind of movie you're looking for, and I'll suggest some great options for you!"}
            ]
            
        if "recommendation_graph" not in st.session_state:
            st.session_state.recommendation_graph = create_recommendation_graph()
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What kind of movie are you looking for?"):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Run the graph
                    result = st.session_state.recommendation_graph.invoke({"query": prompt})
                    
                    # Display recommendations
                    st.write("Here are some movies you might enjoy:")
                    
                    for i, movie in enumerate(result.recommendations, 1):
                        movie_details = []
                        if movie.year:
                            movie_details.append(f"({movie.year})")
                        if movie.genre:
                            movie_details.append(f"**Genre:** {movie.genre}")
                        
                        details_str = " - ".join(movie_details) if movie_details else ""
                        
                        with st.expander(f"{i}. {movie.title} {details_str}"):
                            if movie.description:
                                st.write(f"**Description:** {movie.description}")
                            st.write(f"**Why this might suit you:** {movie.why_recommended}")
                    
                    # Add response to chat history
                    response = "Here are some movie recommendations based on your request. Would you like more specific suggestions or different genres?"
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    create_ui()