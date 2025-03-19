# finance_agent.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import io
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
import json

# Set page config
st.set_page_config(page_title="Personal Finance Manager", layout="wide")

# Initialize Ollama LLM
@st.cache_resource
def get_llm():
    return Ollama(model="llama3.1", temperature=0.1)

# Data Models for LLM Output Parsing
class ExpenseCategory(BaseModel):
    category: str = Field(description="Category of expense")
    amount: float = Field(description="Total amount spent in this category")
    percentage: float = Field(description="Percentage of total expenses")

class FinancialInsight(BaseModel):
    insight: str = Field(description="A specific financial insight")
    impact: str = Field(description="The impact of this insight (positive/negative)")
    recommendation: str = Field(description="Specific recommendation based on this insight")

class FinancialAnalysis(BaseModel):
    summary: str = Field(description="High-level summary of financial situation")
    total_income: float = Field(description="Total income identified")
    total_expenses: float = Field(description="Total expenses identified")
    savings_rate: float = Field(description="Current savings rate as percentage")
    top_expenses: List[ExpenseCategory] = Field(description="Top expense categories")
    insights: List[FinancialInsight] = Field(description="Financial insights")
    recommendations: List[str] = Field(description="Actionable recommendations to improve finances")

# Helper Functions
def detect_file_type(file):
    """Detect the type of file uploaded and read accordingly"""
    filename = file.name.lower()
    if filename.endswith('.csv'):
        return pd.read_csv(file)
    elif filename.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload CSV or Excel files.")
        return None

def analyze_data_structure(df):
    """Analyze data structure and try to identify columns related to finance"""
    # Check column names for common financial terms
    common_date_cols = ['date', 'transaction_date', 'timestamp']
    common_amount_cols = ['amount', 'value', 'transaction', 'price', 'cost', 'payment']
    common_desc_cols = ['description', 'desc', 'transaction_description', 'name', 'details', 'category']
    common_type_cols = ['type', 'transaction_type', 'category']
    
    # Find likely column matches
    date_cols = [col for col in df.columns if any(term in col.lower() for term in common_date_cols)]
    amount_cols = [col for col in df.columns if any(term in col.lower() for term in common_amount_cols)]
    desc_cols = [col for col in df.columns if any(term in col.lower() for term in common_desc_cols)]
    type_cols = [col for col in df.columns if any(term in col.lower() for term in common_type_cols)]
    
    # Identify numeric columns for amount if none found by name
    if not amount_cols:
        amount_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Identify date columns if none found by name
    if not date_cols:
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                continue
                
    return {
        "date_columns": date_cols,
        "amount_columns": amount_cols,
        "description_columns": desc_cols,
        "type_columns": type_cols
    }

def preprocess_financial_data(df, date_col=None, amount_col=None, description_col=None, category_col=None):
    """Preprocess data for analysis"""
    df_processed = df.copy()
    
    # Handle date column
    if date_col:
        df_processed['date'] = pd.to_datetime(df_processed[date_col], errors='coerce')
    
    # Handle amount column
    if amount_col:
        # Clean and convert amount column
        df_processed['amount'] = pd.to_numeric(df_processed[amount_col].astype(str).str.replace('[$,()]', '', regex=True), errors='coerce')
    
    # Handle description column
    if description_col:
        df_processed['description'] = df_processed[description_col].astype(str)
    
    # Handle category column if exists
    if category_col:
        df_processed['category'] = df_processed[category_col].astype(str)
    
    # Drop rows with NaN in critical columns
    critical_cols = [col for col in ['date', 'amount'] if col in df_processed.columns]
    if critical_cols:
        df_processed = df_processed.dropna(subset=critical_cols)
    
    return df_processed

def categorize_transactions(df, description_col="description", llm=None):
    """Use LLM to categorize transactions based on descriptions"""
    if llm is None:
        llm = get_llm()
    
    if 'category' in df.columns and not df['category'].isna().all():
        return df  # Already categorized
    
    # Create a set of unique descriptions to avoid redundant API calls
    unique_descriptions = df[description_col].dropna().unique()
    
    # Template for categorization
    categorization_template = """
    You are a financial expert. Categorize the following transaction description into one of these categories:
    - Housing (rent, mortgage, repairs)
    - Transportation (car payment, gas, public transit)
    - Food (groceries, restaurants)
    - Utilities (electricity, water, internet)
    - Healthcare (doctor visits, medications)
    - Entertainment (movies, games, subscriptions)
    - Shopping (clothing, electronics)
    - Personal Care (haircuts, gym)
    - Education (tuition, books)
    - Travel (hotels, flights)
    - Insurance (health, auto)
    - Debt Payments (credit cards, loans)
    - Savings/Investments (401k, IRA)
    - Income (salary, side gigs)
    - Other

    Transaction description: {description}
    
    Category:
    """
    
    categorization_prompt = PromptTemplate(
        input_variables=["description"],
        template=categorization_template
    )
    
    categorization_chain = LLMChain(
        llm=llm,
        prompt=categorization_prompt
    )
    
    # Create a category mapping dictionary
    category_map = {}
    
    # Set progress bar for categorization
    with st.spinner("Categorizing transactions..."):
        progress_bar = st.progress(0)
        for i, desc in enumerate(unique_descriptions):
            if pd.notna(desc) and desc.strip():
                try:
                    # Get category from LLM
                    category = categorization_chain.run(desc).strip()
                    category_map[desc] = category
                except Exception as e:
                    st.warning(f"Error categorizing '{desc}': {e}")
                    category_map[desc] = "Other"
            else:
                category_map[desc] = "Uncategorized"
            
            # Update progress
            progress_bar.progress((i + 1) / len(unique_descriptions))
    
    # Map categories to dataframe
    df['category'] = df[description_col].map(category_map)
    
    # Fill any missing categories
    df['category'] = df['category'].fillna("Other")
    
    return df

def generate_financial_analysis(df, llm=None):
    """Generate comprehensive financial analysis using LLM"""
    if llm is None:
        llm = get_llm()
    
    # Create a summary of the data
    data_summary = {}
    
    # Time range
    if 'date' in df.columns:
        start_date = df['date'].min().strftime('%Y-%m-%d')
        end_date = df['date'].max().strftime('%Y-%m-%d')
        data_summary['date_range'] = f"{start_date} to {end_date}"
    
    # Income and expenses
    if 'amount' in df.columns and 'category' in df.columns:
        # Consider positive amounts as income, negative as expenses
        income_df = df[df['amount'] > 0]
        expense_df = df[df['amount'] < 0]
        
        data_summary['total_income'] = income_df['amount'].sum()
        data_summary['total_expenses'] = abs(expense_df['amount'].sum())
        
        if data_summary['total_income'] > 0:
            data_summary['savings_rate'] = (data_summary['total_income'] - data_summary['total_expenses']) / data_summary['total_income'] * 100
        else:
            data_summary['savings_rate'] = 0
        
        # Top expense categories
        expense_by_category = expense_df.groupby('category')['amount'].sum().abs().reset_index()
        expense_by_category = expense_by_category.sort_values('amount', ascending=False)
        
        if not expense_by_category.empty:
            data_summary['top_expenses'] = []
            for _, row in expense_by_category.head(5).iterrows():
                percentage = (row['amount'] / data_summary['total_expenses']) * 100
                data_summary['top_expenses'].append({
                    "category": row['category'],
                    "amount": row['amount'],
                    "percentage": percentage
                })
    
    # Template for financial analysis
    analysis_template = """
    You are a professional financial advisor. Based on the provided financial data, analyze the financial situation and provide insights and recommendations.

    Financial Data:
    {data_summary}

    Please provide a comprehensive financial analysis including:
    1. A high-level summary of the financial situation
    2. Key financial insights with their impacts
    3. Actionable recommendations for improving financial health

    Format your response as a JSON object that matches the following schema:
    {schema}
    """
    
    # Use PydanticOutputParser to parse the response
    parser = PydanticOutputParser(pydantic_object=FinancialAnalysis)
    
    analysis_prompt = PromptTemplate(
        input_variables=["data_summary", "schema"],
        template=analysis_template
    )
    
    analysis_chain = LLMChain(
        llm=llm,
        prompt=analysis_prompt
    )
    
    with st.spinner("Generating financial analysis..."):
        try:
            response = analysis_chain.run(
                data_summary=json.dumps(data_summary, indent=2),
                schema=parser.get_format_instructions()
            )
            
            # Clean the response to ensure it's valid JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:-3]  # Remove ```json and ``` markers
            
            # Parse the response
            analysis_result = json.loads(response)
            return analysis_result
        except Exception as e:
            st.error(f"Error generating financial analysis: {e}")
            return None

def visualize_expenses_by_category(df):
    """Create visualizations for expenses by category"""
    if 'amount' not in df.columns or 'category' not in df.columns:
        return None
    
    # Get expense data (negative amounts)
    expense_df = df[df['amount'] < 0].copy()
    expense_df['amount'] = expense_df['amount'].abs()  # Convert to positive for visualization
    
    if expense_df.empty:
        return None
    
    # Group by category
    expenses_by_category = expense_df.groupby('category')['amount'].sum().reset_index()
    expenses_by_category = expenses_by_category.sort_values('amount', ascending=False)
    
    # Create pie chart
    fig = px.pie(
        expenses_by_category, 
        values='amount', 
        names='category',
        title='Expenses by Category',
        hole=0.4,
    )
    
    return fig

def visualize_income_vs_expenses(df):
    """Create visualizations for income vs expenses over time"""
    if 'amount' not in df.columns or 'date' not in df.columns:
        return None
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by month
    df['month'] = df['date'].dt.to_period('M')
    
    # Calculate monthly income and expenses
    monthly_data = df.groupby('month').apply(
        lambda x: pd.Series({
            'income': x[x['amount'] > 0]['amount'].sum(),
            'expenses': x[x['amount'] < 0]['amount'].abs().sum()
        })
    ).reset_index()
    
    # Convert period to string for plotting
    monthly_data['month_str'] = monthly_data['month'].astype(str)
    
    # Create monthly comparison bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_data['month_str'],
        y=monthly_data['income'],
        name='Income',
        marker_color='green'
    ))
    
    fig.add_trace(go.Bar(
        x=monthly_data['month_str'],
        y=monthly_data['expenses'],
        name='Expenses',
        marker_color='red'
    ))
    
    fig.update_layout(
        title='Monthly Income vs. Expenses',
        xaxis_title='Month',
        yaxis_title='Amount',
        barmode='group'
    )
    
    return fig

def visualize_spending_trends(df):
    """Create visualizations for spending trends over time"""
    if 'amount' not in df.columns or 'date' not in df.columns or 'category' not in df.columns:
        return None
    
    # Get expense data (negative amounts)
    expense_df = df[df['amount'] < 0].copy()
    expense_df['amount'] = expense_df['amount'].abs()  # Convert to positive for visualization
    
    if expense_df.empty:
        return None
    
    # Ensure date column is datetime
    expense_df['date'] = pd.to_datetime(expense_df['date'])
    
    # Group by month and category
    expense_df['month'] = expense_df['date'].dt.to_period('M')
    
    # Calculate monthly expenses by category
    monthly_expenses = expense_df.groupby(['month', 'category'])['amount'].sum().reset_index()
    
    # Convert period to string for plotting
    monthly_expenses['month_str'] = monthly_expenses['month'].astype(str)
    
    # Create line chart for top 5 categories
    top_categories = expense_df.groupby('category')['amount'].sum().nlargest(5).index.tolist()
    filtered_data = monthly_expenses[monthly_expenses['category'].isin(top_categories)]
    
    fig = px.line(
        filtered_data,
        x='month_str',
        y='amount',
        color='category',
        title='Monthly Spending Trends (Top 5 Categories)',
        labels={'month_str': 'Month', 'amount': 'Amount Spent'}
    )
    
    return fig

def visualize_savings_rate(df):
    """Create visualization for savings rate over time"""
    if 'amount' not in df.columns or 'date' not in df.columns:
        return None
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by month
    df['month'] = df['date'].dt.to_period('M')
    
    # Calculate monthly income, expenses, and savings rate
    monthly_data = df.groupby('month').apply(
        lambda x: pd.Series({
            'income': x[x['amount'] > 0]['amount'].sum(),
            'expenses': x[x['amount'] < 0]['amount'].abs().sum()
        })
    ).reset_index()
    
    # Calculate savings and savings rate
    monthly_data['savings'] = monthly_data['income'] - monthly_data['expenses']
    monthly_data['savings_rate'] = (monthly_data['savings'] / monthly_data['income'] * 100).round(2)
    
    # Replace infinity and NaN values (in case income is 0)
    monthly_data['savings_rate'] = monthly_data['savings_rate'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Convert period to string for plotting
    monthly_data['month_str'] = monthly_data['month'].astype(str)
    
    # Create the figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_data['month_str'],
        y=monthly_data['savings'],
        name='Savings',
        marker_color=monthly_data['savings'].apply(lambda x: 'green' if x >= 0 else 'red')
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly_data['month_str'],
        y=monthly_data['savings_rate'],
        mode='lines+markers',
        name='Savings Rate (%)',
        yaxis='y2',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Monthly Savings and Savings Rate',
        xaxis_title='Month',
        yaxis_title='Savings Amount',
        yaxis2=dict(
            title='Savings Rate (%)',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
            overlaying='y',
            side='right'
        ),
    )
    
    return fig

# Main Application
def main():
    # Sidebar
    st.sidebar.title("Personal Finance Manager")
    
    # Upload section in sidebar
    st.sidebar.header("Upload Financial Data")
    uploaded_file = st.sidebar.file_uploader("Upload financial data (CSV or Excel)", type=["csv", "xlsx", "xls"])
    
    # Sample data option
    use_sample_data = st.sidebar.checkbox("Use sample data instead")
    
    # Main area
    st.title("Personal Finance Management Agent")
    st.write("Upload your financial data to get insights and recommendations")
    
    # Initialize session state for processed data
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if 'column_mapping' not in st.session_state:
        st.session_state.column_mapping = {}
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Process data when uploaded
    if uploaded_file is not None or use_sample_data:
        # Load data
        if use_sample_data:
            # Create sample financial data
            dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
            np.random.seed(42)  # For reproducibility
            
            # Generate random transactions
            n_transactions = 100
            random_indices = np.random.choice(len(dates), n_transactions)
            
            # Common transaction descriptions
            income_descriptions = ["Salary", "Freelance Work", "Investment Dividend", "Tax Refund", "Interest Income"]
            expense_descriptions = [
                "Rent", "Groceries", "Electricity Bill", "Internet Bill", "Uber Ride", 
                "Restaurant", "Coffee Shop", "Amazon Purchase", "Gas Station", "Phone Bill",
                "Netflix Subscription", "Gym Membership", "Doctor Visit", "Movie Tickets", "Clothing Store"
            ]
            
            # Transaction types
            transaction_types = ["income", "expense"]
            
            # Generate transactions
            transactions = []
            for i in range(n_transactions):
                transaction_type = np.random.choice(transaction_types, p=[0.3, 0.7])  # 30% income, 70% expense
                
                if transaction_type == "income":
                    description = np.random.choice(income_descriptions)
                    amount = np.random.uniform(1000, 5000)
                else:
                    description = np.random.choice(expense_descriptions)
                    amount = -np.random.uniform(10, 1000)  # Negative for expenses
                
                transactions.append({
                    "date": dates[random_indices[i]],
                    "description": description,
                    "amount": round(amount, 2)
                })
            
            df = pd.DataFrame(transactions)
            
            # Assign categories based on descriptions
            category_mapping = {
                "Salary": "Income",
                "Freelance Work": "Income",
                "Investment Dividend": "Income",
                "Tax Refund": "Income",
                "Interest Income": "Income",
                "Rent": "Housing",
                "Groceries": "Food",
                "Electricity Bill": "Utilities",
                "Internet Bill": "Utilities",
                "Uber Ride": "Transportation",
                "Restaurant": "Food",
                "Coffee Shop": "Food",
                "Amazon Purchase": "Shopping",
                "Gas Station": "Transportation",
                "Phone Bill": "Utilities",
                "Netflix Subscription": "Entertainment",
                "Gym Membership": "Personal Care",
                "Doctor Visit": "Healthcare",
                "Movie Tickets": "Entertainment",
                "Clothing Store": "Shopping"
            }
            
            df['category'] = df['description'].map(category_mapping)
            
        else:
            df = detect_file_type(uploaded_file)
            if df is None:
                return
        
        st.success("Data loaded successfully!")
        
        # Show raw data
        with st.expander("Raw Data Preview"):
            st.dataframe(df.head(10))
        
        # Analyze data structure
        column_info = analyze_data_structure(df)
        
        # Column mapping selection
        st.subheader("Column Mapping")
        st.write("Please confirm which columns contain the following information:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox(
                "Date column", 
                options=["None"] + df.columns.tolist(),
                index=0 if not column_info["date_columns"] else df.columns.get_loc(column_info["date_columns"][0]) + 1
            )
            
            amount_col = st.selectbox(
                "Amount column", 
                options=["None"] + df.columns.tolist(),
                index=0 if not column_info["amount_columns"] else df.columns.get_loc(column_info["amount_columns"][0]) + 1
            )
        
        with col2:
            description_col = st.selectbox(
                "Description column", 
                options=["None"] + df.columns.tolist(),
                index=0 if not column_info["description_columns"] else df.columns.get_loc(column_info["description_columns"][0]) + 1
            )
            
            category_col = st.selectbox(
                "Category column (if exists)", 
                options=["None"] + df.columns.tolist(),
                index=0 if not column_info["type_columns"] else df.columns.get_loc(column_info["type_columns"][0]) + 1
            )
        
        # Store column mapping
        st.session_state.column_mapping = {
            "date": None if date_col == "None" else date_col,
            "amount": None if amount_col == "None" else amount_col,
            "description": None if description_col == "None" else description_col,
            "category": None if category_col == "None" else category_col
        }
        
        # Process data button
        if st.button("Process Data"):
            with st.spinner("Processing data..."):
                # Preprocess data based on column mapping
                processed_df = preprocess_financial_data(
                    df,
                    date_col=st.session_state.column_mapping["date"],
                    amount_col=st.session_state.column_mapping["amount"],
                    description_col=st.session_state.column_mapping["description"],
                    category_col=st.session_state.column_mapping["category"]
                )
                
                # Categorize transactions if needed
                if st.session_state.column_mapping["description"] and (
                    not st.session_state.column_mapping["category"] or 
                    "category" not in processed_df.columns or 
                    processed_df["category"].isna().all()
                ):
                    processed_df = categorize_transactions(
                        processed_df, 
                        description_col="description" if "description" in processed_df.columns else st.session_state.column_mapping["description"]
                    )
                
                # Store processed data
                st.session_state.processed_data = processed_df
                
                # Generate financial analysis
                st.session_state.analysis_results = generate_financial_analysis(processed_df)
            
            st.success("Data processed successfully!")
    
    # Display analysis and visualizations if data has been processed
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        # Display tabs for different sections
        tabs = st.tabs(["Overview", "Expense Analysis", "Income Analysis", "Recommendations"])
        
        with tabs[0]:  # Overview tab
            # Financial summary
            st.subheader("Financial Summary")
            
            if st.session_state.analysis_results:
                analysis = st.session_state.analysis_results
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Income", f"${analysis.get('total_income', 0):,.2f}")
                with col2:
                    st.metric("Total Expenses", f"${analysis.get('total_expenses', 0):,.2f}")
                with col3:
                    st.metric("Savings Rate", f"{analysis.get('savings_rate', 0):.2f}%")
                
                # Display summary text
                st.info(analysis.get('summary', 'No summary available'))
                
                # Income vs expenses chart
                income_vs_expenses_chart = visualize_income_vs_expenses(df)
                if income_vs_expenses_chart:
                    st.plotly_chart(income_vs_expenses_chart, use_container_width=True)
                
                # Savings rate chart
                savings_rate_chart = visualize_savings_rate(df)
                if savings_rate_chart:
                    st.plotly_chart(savings_rate_chart, use_container_width=True)
            else:
                st.warning("Analysis results not available. Please process the data.")
        
        with tabs[1]:  # Expense Analysis tab
            st.subheader("Expense Analysis")
            
            # Expense by category chart
            expense_chart = visualize_expenses_by_category(df)
            if expense_chart:
                st.plotly_chart(expense_chart, use_container_width=True)
            
            # Spending trends chart
            spending_trends_chart = visualize_spending_trends(df)
            if spending_trends_chart:
                st.plotly_chart(spending_trends_chart, use_container_width=True)
            
            # Top expenses table
            if st.session_state.analysis_results and 'top_expenses' in st.session_state.analysis_results:
                st.subheader("Top Expense Categories")
                top_expenses = st.session_state.analysis_results['top_expenses']
                
                if top_expenses:
                    expense_data = {
                        "Category": [exp['category'] for exp in top_expenses],
                        "Amount": [f"${exp['amount']:,.2f}" for exp in top_expenses],
                        "Percentage": [f"{exp['percentage']:.2f}%" for exp in top_expenses]
                    }
                    st.table(pd.DataFrame(expense_data))
        
        with tabs[2]:  # Income Analysis tab
            st.subheader("Income Analysis")
            
            # Filter for income transactions
            income_df = df[df['amount'] > 0]
            
            if not income_df.empty and 'date' in income_df.columns:
                # Monthly income chart
                income_df['month'] = pd.to_datetime(income_df['date']).dt.to_period('M')
                monthly_income = income_df.groupby('month')['amount'].sum().reset_index()
                monthly_income['month_str'] = monthly_income['month'].astype(str)
                
                fig = px.bar(
                    monthly_income,
                    x='month_str',
                    y='amount',
                    title='Monthly Income',
                    labels={'month_str': 'Month', 'amount': 'Income Amount'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Income source breakdown if category exists
                if 'category' in income_df.columns:
                    income_by_category = income_df.groupby('category')['amount'].sum().reset_index()
                    income_by_category = income_by_category.sort_values('amount', ascending=False)
                    
                    fig = px.pie(
                        income_by_category,
                        values='amount',
                        names='category',
                        title='Income Sources'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No income data available for analysis.")
        
        with tabs[3]:  # Recommendations tab
            st.subheader("Financial Recommendations")
            
            if st.session_state.analysis_results:
                analysis = st.session_state.analysis_results
                
                # Display insights
                if 'insights' in analysis and analysis['insights']:
                    st.subheader("Key Insights")
                    for i, insight in enumerate(analysis['insights']):
                        with st.expander(f"Insight {i+1}: {insight.get('insight', 'No insight')}"):
                            st.write(f"**Impact:** {insight.get('impact', 'Unknown')}")
                            st.write(f"**Recommendation:** {insight.get('recommendation', 'No recommendation')}")
                
                # Display recommendations
                if 'recommendations' in analysis and analysis['recommendations']:
                    st.subheader("Action Steps")
                    for i, recommendation in enumerate(analysis['recommendations']):
                        st.markdown(f"**{i+1}.** {recommendation}")
                
                # Interactive goal setting
                st.subheader("Set Financial Goals")
                
                goal_type = st.selectbox(
                    "Select Goal Type",
                    ["Savings Rate", "Expense Reduction", "Debt Repayment", "Income Increase"]
                )
                
                if goal_type == "Savings Rate":
                    current_rate = analysis.get('savings_rate', 0)
                    target_rate = st.slider("Target Savings Rate (%)", 0, 50, int(current_rate) + 5)
                    if st.button("Calculate Plan"):
                        with st.spinner("Generating savings plan..."):
                            # Simple calculation for demonstration
                            current_income = analysis.get('total_income', 0)
                            current_expenses = analysis.get('total_expenses', 0)
                            current_savings = current_income - current_expenses
                            current_rate = (current_savings / current_income * 100) if current_income > 0 else 0
                            
                            target_savings = (current_income * target_rate / 100)
                            savings_gap = target_savings - current_savings
                            
                            # Display plan
                            st.subheader("Savings Rate Plan")
                            st.write(f"Current savings rate: {current_rate:.2f}%")
                            st.write(f"Target savings rate: {target_rate:.0f}%")
                            st.write(f"Additional monthly savings needed: ${savings_gap:.2f}")
                            
                            # Generate recommendations using LLM
                            savings_prompt = PromptTemplate(
                                input_variables=["current_rate", "target_rate", "income", "expenses", "savings_gap"],
                                template="""
                                As a financial advisor, provide 3-5 practical steps to increase savings rate from {current_rate:.2f}% to {target_rate:.0f}%.
                                Current monthly income: ${income:.2f}
                                Current monthly expenses: ${expenses:.2f}
                                Savings gap to fill: ${savings_gap:.2f}
                                
                                Provide specific, actionable advice to either reduce expenses or increase income to reach this goal.
                                Format each recommendation as a bullet point.
                                """
                            )
                            
                            savings_chain = LLMChain(
                                llm=get_llm(),
                                prompt=savings_prompt
                            )
                            
                            recommendations = savings_chain.run(
                                current_rate=current_rate,
                                target_rate=target_rate,
                                income=current_income,
                                expenses=current_expenses,
                                savings_gap=savings_gap
                            )
                            
                            st.write("### Recommendations")
                            st.write(recommendations)
                
                elif goal_type == "Expense Reduction":
                    if 'top_expenses' in analysis and analysis['top_expenses']:
                        target_category = st.selectbox(
                            "Select expense category to reduce",
                            options=[exp['category'] for exp in analysis['top_expenses']]
                        )
                        
                        # Find the selected category
                        selected_expense = next((exp for exp in analysis['top_expenses'] if exp['category'] == target_category), None)
                        
                        if selected_expense:
                            current_amount = selected_expense['amount']
                            reduction_percentage = st.slider("Reduction target (%)", 5, 50, 20)
                            target_amount = current_amount * (1 - reduction_percentage/100)
                            
                            if st.button("Generate Reduction Plan"):
                                with st.spinner("Creating expense reduction plan..."):
                                    # Generate recommendations using LLM
                                    reduction_prompt = PromptTemplate(
                                        input_variables=["category", "current_amount", "target_amount", "reduction_percentage"],
                                        template="""
                                        As a financial advisor, provide 3-5 practical ways to reduce spending in the {category} category.
                                        Current monthly spending: ${current_amount:.2f}
                                        Target monthly spending: ${target_amount:.2f} (a {reduction_percentage}% reduction)
                                        
                                        Provide specific, actionable advice that would realistically help someone reduce this expense category.
                                        Format each recommendation as a bullet point.
                                        """
                                    )
                                    
                                    reduction_chain = LLMChain(
                                        llm=get_llm(),
                                        prompt=reduction_prompt
                                    )
                                    
                                    recommendations = reduction_chain.run(
                                        category=target_category,
                                        current_amount=current_amount,
                                        target_amount=target_amount,
                                        reduction_percentage=reduction_percentage
                                    )
                                    
                                    st.write(f"### {target_category} Reduction Plan")
                                    st.write(f"Current monthly spending: ${current_amount:.2f}")
                                    st.write(f"Target monthly spending: ${target_amount:.2f}")
                                    st.write(f"Monthly savings: ${current_amount - target_amount:.2f}")
                                    
                                    st.write("### Recommendations")
                                    st.write(recommendations)
                
                elif goal_type == "Debt Repayment":
                    # Simple debt repayment calculator
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        debt_amount = st.number_input("Total debt amount ($)", min_value=0.0, value=10000.0, step=1000.0)
                        interest_rate = st.number_input("Annual interest rate (%)", min_value=0.0, max_value=30.0, value=18.0, step=0.5)
                    
                    with col2:
                        monthly_payment = st.number_input("Monthly payment ($)", min_value=0.0, value=500.0, step=50.0)
                    
                    if st.button("Calculate Repayment Plan"):
                        with st.spinner("Calculating debt repayment plan..."):
                            # Calculate time to pay off debt
                            monthly_rate = interest_rate / 100 / 12
                            
                            if monthly_rate == 0:
                                months_to_payoff = debt_amount / monthly_payment
                            else:
                                months_to_payoff = -np.log(1 - debt_amount * monthly_rate / monthly_payment) / np.log(1 + monthly_rate)
                            
                            years = int(months_to_payoff // 12)
                            months = int(months_to_payoff % 12)
                            
                            total_payment = monthly_payment * months_to_payoff
                            total_interest = total_payment - debt_amount
                            
                            st.write(f"### Debt Repayment Plan")
                            st.write(f"Time to pay off debt: {years} years and {months} months")
                            st.write(f"Total payment: ${total_payment:.2f}")
                            st.write(f"Total interest paid: ${total_interest:.2f}")
                            
                            # Generate a monthly payment schedule
                            payment_schedule = []
                            remaining_balance = debt_amount
                            month = 1
                            
                            while remaining_balance > 0:
                                interest_payment = remaining_balance * monthly_rate
                                principal_payment = min(monthly_payment - interest_payment, remaining_balance)
                                
                                remaining_balance -= principal_payment
                                
                                payment_schedule.append({
                                    "Month": month,
                                    "Payment": monthly_payment if remaining_balance > 0 else (principal_payment + interest_payment),
                                    "Principal": principal_payment,
                                    "Interest": interest_payment,
                                    "Remaining Balance": remaining_balance
                                })
                                
                                month += 1
                                if month > 360:  # Limit to 30 years
                                    break
                            
                            # Display payment schedule
                            df_schedule = pd.DataFrame(payment_schedule)
                            st.write("### Payment Schedule")
                            st.dataframe(df_schedule)
                            
                            # Generate recommendations using LLM
                            debt_prompt = PromptTemplate(
                                input_variables=["debt_amount", "interest_rate", "monthly_payment", "payoff_time", "total_interest"],
                                template="""
                                As a financial advisor, provide 3-5 practical strategies to accelerate debt repayment and reduce interest costs.
                                Current debt: ${debt_amount:.2f}
                                Interest rate: {interest_rate:.2f}%
                                Monthly payment: ${monthly_payment:.2f}
                                Current payoff timeline: {payoff_time}
                                Total interest to be paid: ${total_interest:.2f}
                                
                                Provide specific, actionable advice that would help someone pay off this debt faster and save on interest.
                                Format each recommendation as a bullet point.
                                """
                            )
                            
                            debt_chain = LLMChain(
                                llm=get_llm(),
                                prompt=debt_prompt
                            )
                            
                            recommendations = debt_chain.run(
                                debt_amount=debt_amount,
                                interest_rate=interest_rate,
                                monthly_payment=monthly_payment,
                                payoff_time=f"{years} years and {months} months",
                                total_interest=total_interest
                            )
                            
                            st.write("### Recommendations")
                            st.write(recommendations)
                
                elif goal_type == "Income Increase":
                    current_income = analysis.get('total_income', 5000)
                    target_increase = st.slider("Target income increase (%)", 5, 50, 20)
                    target_income = current_income * (1 + target_increase/100)
                    
                    if st.button("Generate Income Growth Plan"):
                        with st.spinner("Creating income growth plan..."):
                            # Generate recommendations using LLM
                            income_prompt = PromptTemplate(
                                input_variables=["current_income", "target_income", "increase_percentage"],
                                template="""
                                As a financial advisor, provide 3-5 practical strategies to increase income by {increase_percentage}%.
                                Current monthly income: ${current_income:.2f}
                                Target monthly income: ${target_income:.2f}
                                
                                Consider multiple approaches including career development, side hustles, passive income, and skill building.
                                Provide specific, actionable advice that would realistically help someone increase their income.
                                Format each recommendation as a bullet point.
                                """
                            )
                            
                            income_chain = LLMChain(
                                llm=get_llm(),
                                prompt=income_prompt
                            )
                            
                            recommendations = income_chain.run(
                                current_income=current_income,
                                target_income=target_income,
                                increase_percentage=target_increase
                            )
                            
                            st.write(f"### Income Growth Plan")
                            st.write(f"Current monthly income: ${current_income:.2f}")
                            st.write(f"Target monthly income: ${target_income:.2f}")
                            st.write(f"Monthly increase: ${target_income - current_income:.2f}")
                            
                            st.write("### Recommendations")
                            st.write(recommendations)
            else:
                st.warning("Analysis results not available. Please process the data.")
    
    # Help/About section in sidebar
    with st.sidebar.expander("About This App"):
        st.write("""
        This personal finance management agent helps you analyze your financial data, visualize spending patterns, 
        and get personalized recommendations to improve your financial health.
        
        **Features:**
        - Upload and analyze financial data from CSV or Excel files
        - Automatic categorization of transactions
        - Visual breakdowns of income and expenses
        - Personalized financial insights and recommendations
        - Interactive financial goal planning
        
        **Technologies Used:**
        - Streamlit for the user interface
        - LangChain for AI agent capabilities
        - Ollama with Llama 3.1 for AI processing
        - Pandas for data processing
        - Plotly for interactive visualizations
        """)

if __name__ == "__main__":
    main()