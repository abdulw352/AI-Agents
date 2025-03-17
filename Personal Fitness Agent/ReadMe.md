# AI Fitness Trainer Agent Framework

This project is a robust personal fitness trainer agent framework that analyzes user health data, visualizes fitness metrics, and provides personalized recommendations for improving health and fitness.

## Features

- **User Profile Management**: Create and save your personal fitness profile
- **Data Analysis**: Upload and analyze fitness data from CSV or Excel files
- **Data Visualization**: Interactive charts and graphs to visualize your fitness journey
- **Personalized Recommendations**: Get AI-generated workout plans and dietary advice
- **Interactive Q&A**: Ask questions about your fitness plan and get personalized answers
- **Export Functionality**: Export your fitness plan in different formats

## Requirements

- Python 3.9+
- Ollama with Llama3.1 model installed
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone 
cd ai-fitness-trainer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Install Ollama from [https://ollama.ai/](https://ollama.ai/)

4. Pull the Llama3.1 model:
```bash
ollama pull llama3.1
```

5. Create a data directory:
```bash
mkdir -p data/exports
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Access the app in your browser at `http://localhost:8501`

3. Fill in your profile information and upload your fitness data (or use the sample data)

4. Generate analysis and recommendations

5. Explore your personalized fitness plan and ask questions

## Data Format

The app accepts CSV or Excel files with fitness data. The recommended format includes:

- A date/time column
- Numerical metrics columns (weight, steps, calories, etc.)

Example data columns:
```
Date, Weight (kg), Steps, Calories Burned, Workout Duration (min), Sleep (hours)
```

## Permanent Data Storage

User data is stored in the `data` directory:
- User profile is saved in `data/config.json`
- Uploaded data files are saved in the `data` directory
- Exported plans are saved in `data/exports`

## Framework Components

- **Streamlit**: User interface
- **LangChain & LangGraph**: Agent framework
- **Ollama with Llama3.1**: Local LLM for privacy and performance
- **Pandas & Plotly**: Data processing and visualization

## Extending the Framework

- Add new visualizations in the `visualize_data` function
- Enhance the agent prompts for better analysis and recommendations
- Add new export formats or integration with fitness apps
- Implement user authentication for multi-user support

## License

MIT