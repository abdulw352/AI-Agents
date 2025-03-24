import os
from langchain_openai import ChatOpenAI
from can import process_user_query

def main():
    # Set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key"  # Uncomment and add your key if not set globally
    
    # Initialize the language model
    llm = ChatOpenAI(temperature=0.7)
    
    # Example queries to test
    example_queries = [
        "What are the latest advancements in quantum computing?",
        "Compare and contrast different renewable energy sources.",
        "Explain the impact of artificial intelligence on healthcare."
    ]
    
    # Process a user query
    print("Select a query to process:")
    for i, query in enumerate(example_queries):
        print(f"{i+1}. {query}")
    print("4. Enter your own query")
    
    choice = int(input("Enter your choice (1-4): "))
    
    if choice == 4:
        query = input("Enter your query: ")
    else:
        query = example_queries[choice-1]
    
    # Set maximum rounds for refinement
    max_rounds = int(input("Enter maximum refinement rounds (1-5): ") or "3")
    
    print(f"\nProcessing query: {query}")
    print(f"Maximum refinement rounds: {max_rounds}")
    print("\nThis may take a few minutes depending on the complexity of the query...\n")
    
    # Process the query
    response = process_user_query(query, llm, max_rounds=max_rounds)
    
    print("=" * 80)
    print("FINAL RESPONSE:")
    print("=" * 80)
    print(response)

if __name__ == "__main__":
    main()