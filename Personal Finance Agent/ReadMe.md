# 💸 Personal Finance Management Agent

A comprehensive AI-powered financial analysis tool that helps users understand their spending patterns, track their financial health, and receive personalized recommendations to improve their financial situation.

## 📊 Overview

The Personal Finance Management Agent is an intelligent financial assistant that combines data analysis with AI-powered insights. It's designed to function as a personal financial advisor by:

- Analyzing financial transaction data from various sources
- Visualizing spending patterns and financial trends
- Providing personalized recommendations and action plans
- Helping users set and achieve financial goals

This application bridges the gap between raw financial data and actionable financial insights, making it easier for individuals to make informed decisions about their money.

## ✨ Features

### Data Processing
- **Multi-format support**: Import data from CSV and Excel files
- **Intelligent column detection**: Automatically identifies date, amount, and transaction fields
- **Smart categorization**: Uses AI to categorize transactions into meaningful groups

### Financial Analysis
- **Comprehensive metrics**: Calculates income, expenses, savings rate, and more
- **Trend identification**: Detects patterns in spending and saving behavior
- **AI-powered insights**: Generates personalized observations about financial health

### Interactive Visualizations
- **Expense breakdown**: Visual representation of spending by category
- **Income vs. expenses**: Track the balance between money in and money out
- **Spending trends**: Monitor how expenses evolve over time
- **Savings tracking**: Visualize savings rate and progress

### Smart Recommendations
- **Personalized advice**: Tailored financial guidance based on actual spending patterns
- **Goal-based planning**: Custom plans for:
  - Increasing savings rate
  - Reducing expenses in specific categories
  - Accelerating debt repayment
  - Growing income

## 🛠️ Technology Stack

- **Framework**: Streamlit for the interactive web interface
- **AI/ML**: LangChain for orchestrating AI workflows with Ollama/Llama 3.1
- **Data Processing**: Pandas for powerful data manipulation
- **Visualization**: Plotly for dynamic, interactive charts
- **Architecture**: Modular design pattern with separation of concerns

## 📋 Requirements

- Python 3.9+
- Ollama (with Llama 3.1 model)
- Dependencies listed in `requirements.txt`

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone 
cd personal-finance-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Ollama and Llama 3.1

```bash
# Install Ollama following instructions at https://ollama.com/

# Once installed, pull the Llama 3.1 model
ollama pull llama3.1
```

### 5. Run the application

```bash
streamlit run app.py
```

The application should now be accessible at `http://localhost:8501`.

## 📝 Usage Guide

### Importing Data

1. **Upload your financial data**:
   - Supported formats: CSV, Excel (.xlsx, .xls)
   - Required columns: Date, Amount, Description (Category is optional)
   - For testing, you can use the "Sample Data" option

2. **Confirm column mappings**:
   - The app will attempt to detect the appropriate columns
   - Verify and adjust the mappings if needed

3. **Process the data**:
   - Click "Process Data" to begin analysis
   - The app will categorize transactions and generate insights

### Exploring Analysis

The application is organized into four main tabs:

1. **Overview**:
   - Financial summary with key metrics
   - Income vs. expenses visualization
   - Savings rate tracking

2. **Expense Analysis**:
   - Breakdown of expenses by category
   - Spending trends over time
   - Top expense categories

3. **Income Analysis**:
   - Monthly income tracking
   - Income source breakdown (if data available)

4. **Recommendations**:
   - AI-generated insights about your financial situation
   - Actionable recommendations to improve financial health
   - Interactive goal-setting tools

### Setting Financial Goals

1. **Select a goal type**:
   - Savings Rate
   - Expense Reduction
   - Debt Repayment
   - Income Increase

2. **Configure your target**:
   - Set parameters specific to your goal

3. **Generate a personalized plan**:
   - Get AI-powered recommendations to achieve your goal
   - View detailed breakdowns and action steps

## 🏗️ Architecture

The application follows a modular design with several key components:

### Data Processing Layer
- `detect_file_type()`: Identifies and reads different file formats
- `analyze_data_structure()`: Intelligently detects column types
- `preprocess_financial_data()`: Cleans and standardizes financial data
- `categorize_transactions()`: Uses LLM to categorize spending

### Analysis Layer
- `generate_financial_analysis()`: Creates comprehensive financial analysis
- Various visualization functions that transform data into interactive charts

### AI Integration Layer
- LangChain chains for different analytical tasks
- Pydantic models for structured LLM outputs
- Prompt templates designed for financial analysis

### UI Layer
- Streamlit components organized for intuitive user experience
- Interactive elements for data exploration and goal setting

## 🧩 Key Components

### 1. Data Ingestion
The application supports multiple data formats and intelligently maps columns to standardized fields.

### 2. Transaction Categorization
The LLM categorizes transactions into standard groups like Housing, Food, Transportation, etc., enabling meaningful analysis.

### 3. Financial Analysis Engine
Combines traditional data analysis with AI insights to generate a comprehensive view of financial health.

### 4. Recommendation System
Provides personalized financial advice based on the specific patterns and challenges identified in your data.

### 5. Goal Planning Tools
Interactive tools that help users set financial goals and generate concrete plans to achieve them.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🌐 Contact

Project Link: [https://github.com/abdulw352/AI-Agents/personal-finance-agent](https://github.com/abdulw352/AI-Agents/personal-finance-agent)

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the AI orchestration framework
- [Ollama](https://github.com/ollama/ollama) for local LLM hosting
- [Streamlit](https://streamlit.io/) for the interactive web framework
- [Plotly](https://plotly.com/) for the visualization library