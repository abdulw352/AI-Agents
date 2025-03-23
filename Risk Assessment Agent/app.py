import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
from typing import Dict, List, Any, TypedDict, Annotated, Sequence
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from duckduckgo_search import DDGS
import requests
from crawl4ai.functions import crawl
import json
import time

# Configure the LLM
llm = Ollama(model="llama3.1")

# Define state schema for our system
class AgentState(TypedDict):
    portfolio_data: Dict
    market_analysis: Dict
    macro_analysis: Dict
    sector_analysis: Dict
    stock_analysis: Dict
    risk_assessment: Dict
    recommendations: Dict
    errors: List[str]
    final_report: Dict

# Function to parse portfolio from uploaded file
def parse_portfolio(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload CSV or Excel."
        
        # Validate required columns: 'ticker', 'shares' or 'value'
        required_cols = ['ticker']
        optional_cols = ['shares', 'value', 'entry_date', 'cost_basis']
        
        if not all(col in df.columns for col in required_cols):
            return None, f"Portfolio must contain the following columns: {', '.join(required_cols)}"
        
        if not any(col in df.columns for col in optional_cols):
            return None, f"Portfolio must contain at least one of: {', '.join(optional_cols)}"
        
        # Standardize column names and clean up data
        df['ticker'] = df['ticker'].str.upper().str.strip()
        
        # If we have shares but no value, calculate value
        if 'shares' in df.columns and 'value' not in df.columns:
            # Get current prices
            tickers = df['ticker'].unique().tolist()
            prices = {}
            for ticker in tickers:
                try:
                    ticker_data = yf.Ticker(ticker)
                    current_price = ticker_data.info.get('regularMarketPrice', None)
                    if current_price:
                        prices[ticker] = current_price
                    else:
                        prices[ticker] = ticker_data.history(period='1d')['Close'].iloc[-1]
                except Exception as e:
                    return None, f"Error fetching price for {ticker}: {str(e)}"
            
            df['value'] = df.apply(lambda row: row['shares'] * prices[row['ticker']], axis=1)
        
        # Calculate portfolio percentages
        total_value = df['value'].sum()
        df['weight'] = df['value'] / total_value
        
        # Return as dictionary for easier handling
        portfolio_dict = {
            'tickers': df['ticker'].tolist(),
            'weights': df['weight'].tolist(),
            'total_value': float(total_value),
            'holdings': df.to_dict('records'),
            'raw_df': df
        }
        
        return portfolio_dict, None
    
    except Exception as e:
        return None, f"Error parsing portfolio: {str(e)}"

# Agent functions

def market_analysis_agent(state: AgentState) -> AgentState:
    """Analyze overall market conditions, trends, and volatility."""
    try:
        portfolio = state["portfolio_data"]
        tickers = portfolio["tickers"]
        
        # Get market index data
        indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']  # S&P 500, Dow Jones, Nasdaq, VIX
        market_data = {}
        
        for index in indices:
            data = yf.Ticker(index)
            hist = data.history(period="6mo")
            
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                prev_month = hist['Close'].iloc[-22] if len(hist) >= 22 else hist['Close'].iloc[0]
                prev_3month = hist['Close'].iloc[-66] if len(hist) >= 66 else hist['Close'].iloc[0]
                
                market_data[index] = {
                    "current": float(current),
                    "1m_change": float((current / prev_month - 1) * 100),
                    "3m_change": float((current / prev_3month - 1) * 100),
                    "volatility_30d": float(hist['Close'].pct_change().rolling(30).std().iloc[-1] * 100) if len(hist) >= 30 else None
                }
        
        # Get recent market news
        with DDGS() as ddgs:
            market_news = list(ddgs.news(
                keywords="stock market analysis Federal Reserve inflation interest rates",
                region="us",
                time="m",
                max_results=5
            ))
        
        # Prepare market analysis using LLM
        market_prompt = f"""
        You are a market analysis expert. Based on the following market data, provide a concise analysis of current market conditions, trends, and potential risks.
        
        MARKET DATA:
        S&P 500: Current {market_data.get('^GSPC', {}).get('current')}, 1-month change: {market_data.get('^GSPC', {}).get('1m_change')}%, 3-month change: {market_data.get('^GSPC', {}).get('3m_change')}%, 30-day volatility: {market_data.get('^GSPC', {}).get('volatility_30d')}%
        Dow Jones: Current {market_data.get('^DJI', {}).get('current')}, 1-month change: {market_data.get('^DJI', {}).get('1m_change')}%, 3-month change: {market_data.get('^DJI', {}).get('3m_change')}%, 30-day volatility: {market_data.get('^DJI', {}).get('volatility_30d')}%
        Nasdaq: Current {market_data.get('^IXIC', {}).get('current')}, 1-month change: {market_data.get('^IXIC', {}).get('1m_change')}%, 3-month change: {market_data.get('^IXIC', {}).get('3m_change')}%, 30-day volatility: {market_data.get('^IXIC', {}).get('volatility_30d')}%
        VIX: Current {market_data.get('^VIX', {}).get('current')}, 1-month change: {market_data.get('^VIX', {}).get('1m_change')}%, 3-month change: {market_data.get('^VIX', {}).get('3m_change')}%
        
        RECENT NEWS HEADLINES:
        {json.dumps([news.get('title') for news in market_news], indent=2)}
        
        Analyze the current market environment, identify major trends, and assess overall market risk levels (low, moderate, high, or severe).
        Focus on what this means for equity investors in the US market. Your analysis should include:
        1. Current market sentiment
        2. Major risk factors
        3. Overall market risk level with justification
        4. Any protective measures that might be appropriate
        
        Provide your analysis in JSON format with these keys: sentiment, risk_factors, risk_level, protective_measures, summary.
        """
        
        market_analysis_response = llm.invoke(market_prompt)
        
        # Extract the JSON from the response
        try:
            # Find JSON content between triple backticks if present
            if "```json" in market_analysis_response:
                json_str = market_analysis_response.split("```json")[1].split("```")[0].strip()
            elif "```" in market_analysis_response:
                json_str = market_analysis_response.split("```")[1].strip()
            else:
                json_str = market_analysis_response.strip()
                
            market_analysis = json.loads(json_str)
        except:
            # If JSON parsing fails, create a structured response
            market_analysis = {
                "sentiment": "Unable to parse market sentiment",
                "risk_factors": ["Data parsing error"],
                "risk_level": "unknown",
                "protective_measures": ["Consult a financial advisor"],
                "summary": "There was an error analyzing the market data. Please review the raw figures."
            }
        
        # Add raw data for dashboard
        market_analysis["raw_data"] = market_data
        market_analysis["news"] = market_news
        
        state["market_analysis"] = market_analysis
        return state
    
    except Exception as e:
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Market analysis error: {str(e)}")
        state["market_analysis"] = {"error": str(e)}
        return state

def macro_analysis_agent(state: AgentState) -> AgentState:
    """Analyze macroeconomic factors affecting the portfolio."""
    try:
        # Get key macroeconomic indicators
        macro_indicators = {}
        
        # Use DuckDuckGo search to find recent data
        with DDGS() as ddgs:
            inflation_search = list(ddgs.text(
                "current US inflation rate CPI Federal Reserve",
                region="us",
                max_results=3
            ))
            
            interest_rate_search = list(ddgs.text(
                "current Federal Reserve interest rate Fed funds rate",
                region="us",
                max_results=3
            ))
            
            gdp_search = list(ddgs.text(
                "US GDP growth rate latest quarter",
                region="us",
                max_results=3
            ))
            
            unemployment_search = list(ddgs.text(
                "current US unemployment rate",
                region="us",
                max_results=3
            ))
        
        # Use crawl4ai to get Fed minutes or statements
        try:
            fed_data = crawl("https://www.federalreserve.gov/newsevents/pressreleases.htm")
        except:
            fed_data = "Unable to retrieve Federal Reserve information."
        
        # Prepare macroeconomic analysis using LLM
        macro_prompt = f"""
        You are a macroeconomic analyst specializing in financial markets. Based on the following information, provide a concise analysis of current macroeconomic conditions and their impact on US equity markets.
        
        SEARCH RESULTS ABOUT INFLATION:
        {json.dumps(inflation_search, indent=2)}
        
        SEARCH RESULTS ABOUT INTEREST RATES:
        {json.dumps(interest_rate_search, indent=2)}
        
        SEARCH RESULTS ABOUT GDP:
        {json.dumps(gdp_search, indent=2)}
        
        SEARCH RESULTS ABOUT UNEMPLOYMENT:
        {json.dumps(unemployment_search, indent=2)}
        
        FEDERAL RESERVE INFORMATION:
        {fed_data[:2000]}
        
        Extract and analyze:
        1. Current inflation rate and trend
        2. Current interest rates and expected Fed policy
        3. GDP growth outlook
        4. Employment situation
        5. Overall macroeconomic risk level (low, moderate, high, severe)
        
        Also identify which economic factors would most impact equity investments in the current environment.
        
        Provide your analysis in JSON format with these keys: inflation, interest_rates, gdp, employment, risk_level, key_factors, summary.
        """
        
        macro_analysis_response = llm.invoke(macro_prompt)
        
        # Extract the JSON from the response
        try:
            # Find JSON content between triple backticks if present
            if "```json" in macro_analysis_response:
                json_str = macro_analysis_response.split("```json")[1].split("```")[0].strip()
            elif "```" in macro_analysis_response:
                json_str = macro_analysis_response.split("```")[1].strip()
            else:
                json_str = macro_analysis_response.strip()
                
            macro_analysis = json.loads(json_str)
        except:
            # If JSON parsing fails, create a structured response
            macro_analysis = {
                "inflation": "Unable to parse inflation data",
                "interest_rates": "Unable to parse interest rate data",
                "gdp": "Unable to parse GDP data",
                "employment": "Unable to parse employment data",
                "risk_level": "unknown",
                "key_factors": ["Data parsing error"],
                "summary": "There was an error analyzing the macroeconomic data."
            }
        
        # Add raw data
        macro_analysis["raw_search_data"] = {
            "inflation": inflation_search,
            "interest_rates": interest_rate_search,
            "gdp": gdp_search,
            "unemployment": unemployment_search
        }
        
        state["macro_analysis"] = macro_analysis
        return state
    
    except Exception as e:
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Macroeconomic analysis error: {str(e)}")
        state["macro_analysis"] = {"error": str(e)}
        return state

def sector_analysis_agent(state: AgentState) -> AgentState:
    """Analyze the sectors represented in the portfolio."""
    try:
        portfolio = state["portfolio_data"]
        tickers = portfolio["tickers"]
        
        # Get sector information for each ticker
        sector_data = {}
        sector_allocations = {}
        sector_performance = {}
        
        # Fetch sector data
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                sector = stock.info.get('sector', 'Unknown')
                industry = stock.info.get('industry', 'Unknown')
                
                if sector not in sector_data:
                    sector_data[sector] = []
                
                # Find the weight of this stock
                weight = next((item['weight'] for item in portfolio['holdings'] if item['ticker'] == ticker), 0)
                
                sector_data[sector].append({
                    'ticker': ticker,
                    'industry': industry,
                    'weight': weight
                })
                
                # Add to sector allocation
                if sector in sector_allocations:
                    sector_allocations[sector] += weight
                else:
                    sector_allocations[sector] = weight
            except:
                # If we can't get sector info, categorize as Unknown
                if 'Unknown' not in sector_data:
                    sector_data['Unknown'] = []
                
                weight = next((item['weight'] for item in portfolio['holdings'] if item['ticker'] == ticker), 0)
                
                sector_data['Unknown'].append({
                    'ticker': ticker,
                    'industry': 'Unknown',
                    'weight': weight
                })
                
                if 'Unknown' in sector_allocations:
                    sector_allocations['Unknown'] += weight
                else:
                    sector_allocations['Unknown'] = weight
        
        # Get sector ETF performance for common sectors
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Materials': 'XLB',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Communication Services': 'XLC'
        }
        
        for sector, etf in sector_etfs.items():
            try:
                etf_data = yf.Ticker(etf)
                hist = etf_data.history(period="3mo")
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev_month = hist['Close'].iloc[-22] if len(hist) >= 22 else hist['Close'].iloc[0]
                    prev_3month = hist['Close'].iloc[-66] if len(hist) >= 66 else hist['Close'].iloc[0]
                    
                    sector_performance[sector] = {
                        "etf": etf,
                        "1m_change": float((current / prev_month - 1) * 100),
                        "3m_change": float((current / prev_3month - 1) * 100),
                        "volatility_30d": float(hist['Close'].pct_change().rolling(30).std().iloc[-1] * 100) if len(hist) >= 30 else None
                    }
            except:
                continue
        
        # For sectors with high allocations, get news
        significant_sectors = [s for s, w in sector_allocations.items() if w > 0.1 and s != 'Unknown']
        sector_news = {}
        
        with DDGS() as ddgs:
            for sector in significant_sectors[:3]:  # Limit to top 3 sectors
                news = list(ddgs.news(
                    keywords=f"{sector} sector stock market outlook",
                    region="us",
                    time="m",
                    max_results=3
                ))
                sector_news[sector] = news
        
        # Prepare sector analysis using LLM
        sector_prompt = f"""
        You are a sector analyst specializing in equity markets. Based on the following portfolio sector allocation and performance data, provide a detailed analysis of the portfolio's sector exposures, risks, and opportunities.
        
        SECTOR ALLOCATIONS:
        {json.dumps(sector_allocations, indent=2)}
        
        SECTOR PERFORMANCE (3-month):
        {json.dumps(sector_performance, indent=2)}
        
        SECTOR NEWS HEADLINES:
        {json.dumps({s: [n.get('title') for n in news] for s, news in sector_news.items()}, indent=2)}
        
        Analyze:
        1. Concentration risk in specific sectors
        2. Performance outlook for the portfolio's largest sector exposures
        3. Sector-specific risks that might affect the portfolio
        4. Recommendations for sector rebalancing (if needed)
        5. Overall sector risk level (low, moderate, high, severe)
        
        Provide your analysis in JSON format with these keys: concentration_risk, sector_outlook, specific_risks, rebalancing_recommendations, risk_level, summary.
        """
        
        sector_analysis_response = llm.invoke(sector_prompt)
        
        # Extract the JSON from the response
        try:
            # Find JSON content between triple backticks if present
            if "```json" in sector_analysis_response:
                json_str = sector_analysis_response.split("```json")[1].split("```")[0].strip()
            elif "```" in sector_analysis_response:
                json_str = sector_analysis_response.split("```")[1].strip()
            else:
                json_str = sector_analysis_response.strip()
                
            sector_analysis = json.loads(json_str)
        except:
            # If JSON parsing fails, create a structured response
            sector_analysis = {
                "concentration_risk": "Unable to analyze concentration risk",
                "sector_outlook": "Unable to analyze sector outlook",
                "specific_risks": ["Data parsing error"],
                "rebalancing_recommendations": ["Consult a financial advisor"],
                "risk_level": "unknown",
                "summary": "There was an error analyzing the sector data."
            }
        
        # Add raw data
        sector_analysis["raw_data"] = {
            "sector_data": sector_data,
            "sector_allocations": sector_allocations,
            "sector_performance": sector_performance,
            "sector_news": sector_news
        }
        
        state["sector_analysis"] = sector_analysis
        return state
    
    except Exception as e:
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Sector analysis error: {str(e)}")
        state["sector_analysis"] = {"error": str(e)}
        return state

def stock_analysis_agent(state: AgentState) -> AgentState:
    """Analyze individual stocks in the portfolio."""
    try:
        portfolio = state["portfolio_data"]
        tickers = portfolio["tickers"]
        
        # Get stock data for each ticker
        stock_data = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="6mo")
                
                if hist.empty:
                    continue
                
                # Calculate key metrics
                current_price = hist['Close'].iloc[-1]
                prev_month = hist['Close'].iloc[-22] if len(hist) >= 22 else hist['Close'].iloc[0]
                prev_3month = hist['Close'].iloc[-66] if len(hist) >= 66 else hist['Close'].iloc[0]
                
                # Get basic info
                info = {
                    "name": stock.info.get('shortName', ticker),
                    "sector": stock.info.get('sector', 'Unknown'),
                    "industry": stock.info.get('industry', 'Unknown'),
                    "market_cap": stock.info.get('marketCap', None),
                    "pe_ratio": stock.info.get('trailingPE', None),
                    "dividend_yield": stock.info.get('dividendYield', None),
                    "beta": stock.info.get('beta', None),
                    "52w_high": stock.info.get('fiftyTwoWeekHigh', None),
                    "52w_low": stock.info.get('fiftyTwoWeekLow', None),
                    "current_price": float(current_price),
                    "1m_change": float((current_price / prev_month - 1) * 100),
                    "3m_change": float((current_price / prev_3month - 1) * 100),
                    "volatility_30d": float(hist['Close'].pct_change().rolling(30).std().iloc[-1] * 100) if len(hist) >= 30 else None
                }
                
                # Get analyst recommendations if available
                try:
                    recommendations = stock.recommendations
                    if not recommendations.empty:
                        latest_recs = recommendations.tail(5)
                        info["analyst_recommendations"] = latest_recs.to_dict('records')
                except:
                    info["analyst_recommendations"] = []
                
                # Find weight in portfolio
                weight = next((item['weight'] for item in portfolio['holdings'] if item['ticker'] == ticker), 0)
                info["portfolio_weight"] = weight
                
                stock_data[ticker] = info
                
            except Exception as e:
                # If we can't get stock info, add basic error info
                stock_data[ticker] = {
                    "name": ticker,
                    "error": str(e),
                    "portfolio_weight": next((item['weight'] for item in portfolio['holdings'] if item['ticker'] == ticker), 0)
                }
        
        # Get news for the top holdings (by weight)
        top_holdings = sorted([(t, d.get('portfolio_weight', 0)) 
                              for t, d in stock_data.items()], 
                              key=lambda x: x[1], reverse=True)[:5]
        
        stock_news = {}
        with DDGS() as ddgs:
            for ticker, _ in top_holdings:
                company_name = stock_data[ticker].get('name', ticker)
                news = list(ddgs.news(
                    keywords=f"{company_name} stock {ticker} financial news",
                    region="us",
                    time="w",
                    max_results=3
                ))
                stock_news[ticker] = news
        
        # Analyze highest risk stocks
        volatile_stocks = sorted([(t, d.get('volatility_30d', 0)) 
                                 for t, d in stock_data.items() if d.get('volatility_30d') is not None], 
                                 key=lambda x: x[1], reverse=True)[:3]
        
        # Prepare stock analysis using LLM
        stock_prompt = f"""
        You are a stock analyst and portfolio manager. Based on the following information about stocks in a portfolio, provide a detailed risk analysis focusing on individual securities.
        
        TOP HOLDINGS:
        {json.dumps([{"ticker": t, "weight": w, "name": stock_data[t].get('name', t)} for t, w in top_holdings], indent=2)}
        
        MOST VOLATILE HOLDINGS:
        {json.dumps([{"ticker": t, "volatility_30d": v, "name": stock_data[t].get('name', t)} for t, v in volatile_stocks], indent=2)}
        
        RECENT NEWS HEADLINES FOR TOP HOLDINGS:
        {json.dumps({t: [n.get('title') for n in stock_news.get(t, [])] for t, _ in top_holdings}, indent=2)}
        
        Analyze:
        1. Single-stock concentration risk (identify stocks with excessive weight)
        2. Stocks with concerning fundamentals or technical indicators
        3. Stocks with notable recent news that might affect performance
        4. High volatility securities that may need risk management
        5. Overall single-stock risk level (low, moderate, high, severe)
        
        Provide your analysis in JSON format with these keys: concentration_risk, concerning_fundamentals, news_impact, volatility_concerns, risk_level, summary.
        """
        
        stock_analysis_response = llm.invoke(stock_prompt)
        
        # Extract the JSON from the response
        try:
            # Find JSON content between triple backticks if present
            if "```json" in stock_analysis_response:
                json_str = stock_analysis_response.split("```json")[1].split("```")[0].strip()
            elif "```" in stock_analysis_response:
                json_str = stock_analysis_response.split("```")[1].strip()
            else:
                json_str = stock_analysis_response.strip()
                
            stock_analysis = json.loads(json_str)
        except:
            # If JSON parsing fails, create a structured response
            stock_analysis = {
                "concentration_risk": "Unable to analyze concentration risk",
                "concerning_fundamentals": ["Data parsing error"],
                "news_impact": "Unable to analyze news impact",
                "volatility_concerns": ["Unable to analyze volatility"],
                "risk_level": "unknown",
                "summary": "There was an error analyzing the stock data."
            }
        
        # Add raw data
        stock_analysis["raw_data"] = {
            "stock_data": stock_data,
            "top_holdings": top_holdings,
            "volatile_stocks": volatile_stocks,
            "stock_news": stock_news
        }
        
        state["stock_analysis"] = stock_analysis
        return state
    
    except Exception as e:
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Stock analysis error: {str(e)}")
        state["stock_analysis"] = {"error": str(e)}
        return state

def risk_assessment_agent(state: AgentState) -> AgentState:
    """Consolidate analyses and perform comprehensive risk assessment."""
    try:
        portfolio = state["portfolio_data"]
        market = state["market_analysis"]
        macro = state["macro_analysis"]
        sector = state["sector_analysis"]
        stock = state["stock_analysis"]
        
        # Calculate portfolio statistics
        tickers = portfolio["tickers"]
        weights = portfolio["weights"]
        
        # Get historical prices for correlation and volatility
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        try:
            # Download all price data at once
            prices_df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
            
            # Calculate returns
            returns_df = prices_df.pct_change().dropna()
            
            # Portfolio volatility
            portfolio_returns = returns_df.dot(weights)
            portfolio_volatility = portfolio_returns.std() * (252 ** 0.5) * 100  # Annualized, in percentage
            
            # VaR calculation (95% confidence)
            var_95 = np.percentile(portfolio_returns, 5) * portfolio["total_value"]
            
            # Expected shortfall/CVaR
            cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * portfolio["total_value"]
            
            # Maximum drawdown
            cum_returns = (1 + portfolio_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns / running_max - 1) * 100
            max_drawdown = drawdown.min()
            
            # Correlation matrix (for top holdings)
            correlation_matrix = {}
            if len(returns_df.columns) > 1:
                corr_matrix = returns_df.corr()
                
                # Only include top holdings for clarity
                top_n = min(5, len(tickers))
                top_tickers = [t for t, w in sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True)][:top_n]
                
                for i, ticker1 in enumerate(top_tickers):
                    correlation_matrix[ticker1] = {}
                    for ticker2 in top_tickers:
                        if ticker1 in corr_matrix.index and ticker2 in corr_matrix.columns:
                            correlation_matrix[ticker1][ticker2] = float(corr_matrix.loc[ticker1, ticker2])
            
            # Risk metrics
            risk_metrics = {
                "annualized_volatility": float(portfolio_volatility),
                "var_95": float(var_95),
                "cvar_95": float(cvar_95),
                "max_drawdown": float(max_drawdown),
                "correlation_matrix": correlation_matrix
            }
        except Exception as e:
            # If historical analysis fails, create placeholder data
            risk_metrics = {
                "annualized_volatility": "Unable to calculate",
                "var_95": "Unable to calculate",
                "cvar_95": "Unable to calculate",
                "max_drawdown": "Unable to calculate",
                "correlation_matrix": {},
                "error": str(e)
            }
        
        # Prepare comprehensive risk assessment using LLM
        risk_prompt = f"""
        You are an expert risk manager for investment portfolios. Based on the comprehensive analyses provided, create a detailed risk assessment for this portfolio.
        
        PORTFOLIO RISK METRICS:
        {json.dumps(risk_metrics, indent=2)}
        
        MARKET ANALYSIS SUMMARY:
        Risk Level: {market.get('risk_level', 'unknown')}
        {market.get('summary', 'No market analysis available')}
        
        MACROECONOMIC ANALYSIS SUMMARY:
        Risk Level: {macro.get('risk_level', 'unknown')}
        {macro.get('summary', 'No macroeconomic analysis available')}
        
        SECTOR ANALYSIS SUMMARY:
        Risk Level: {sector.get('risk_level', 'unknown')}
        {sector.get('summary', 'No sector analysis available')}
        
        STOCK ANALYSIS SUMMARY:
        Risk Level: {stock.get('risk_level', 'unknown')}
        {stock.get('summary', 'No stock analysis available')}
        
        Based on all analyses, provide a comprehensive risk assessment including:
        1. Overall portfolio risk level (low, moderate, high, severe)
        2. Key risk factors categorized by market, macro, sector, and stock-specific risks
        3. Risk metrics interpretation        
        4. Correlation and diversification assessment
        5. Portfolio vulnerabilities under different market scenarios
        6. Risk prioritization (which risks are most critical to address)
        
        Provide your assessment in JSON format with these keys: overall_risk_level, key_risk_factors, metrics_interpretation, diversification_assessment, scenario_vulnerabilities, risk_priorities, summary.
        """
        
        risk_assessment_response = llm.invoke(risk_prompt)
        
        # Extract the JSON from the response
        try:
            # Find JSON content between triple backticks if present
            if "```json" in risk_assessment_response:
                json_str = risk_assessment_response.split("```json")[1].split("```")[0].strip()
            elif "```" in risk_assessment_response:
                json_str = risk_assessment_response.split("```")[1].strip()
            else:
                json_str = risk_assessment_response.strip()
                
            risk_assessment = json.loads(json_str)
        except:
            # If JSON parsing fails, create a structured response
            risk_assessment = {
                "overall_risk_level": "unknown",
                "key_risk_factors": ["Data parsing error"],
                "metrics_interpretation": "Unable to interpret metrics",
                "diversification_assessment": "Unable to assess diversification",
                "scenario_vulnerabilities": ["Unable to analyze vulnerabilities"],
                "risk_priorities": ["Consult a financial advisor"],
                "summary": "There was an error compiling the risk assessment."
            }
        
        # Add raw metrics
        risk_assessment["raw_metrics"] = risk_metrics
        
        state["risk_assessment"] = risk_assessment
        return state
    
    except Exception as e:
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Risk assessment error: {str(e)}")
        state["risk_assessment"] = {"error": str(e)}
        return state

def recommendations_agent(state: AgentState) -> AgentState:
    """Generate actionable recommendations based on risk assessment."""
    try:
        portfolio = state["portfolio_data"]
        risk_assessment = state["risk_assessment"]
        
        # Prepare recommendations using LLM
        recommendations_prompt = f"""
        You are a fiduciary financial advisor specializing in portfolio risk management. Based on the comprehensive risk assessment provided, generate actionable recommendations to improve the portfolio's risk profile.
        
        PORTFOLIO OVERVIEW:
        Total Value: ${portfolio.get('total_value', 0):,.2f}
        Number of Holdings: {len(portfolio.get('tickers', []))}
        
        RISK ASSESSMENT:
        Overall Risk Level: {risk_assessment.get('overall_risk_level', 'unknown')}
        Key Risk Factors: {json.dumps(risk_assessment.get('key_risk_factors', []), indent=2)}
        Risk Priorities: {json.dumps(risk_assessment.get('risk_priorities', []), indent=2)}
        
        Based on this assessment, provide specific, actionable recommendations to manage the identified risks. Your recommendations should:
        1. Address the highest priority risks first
        2. Include specific actions (e.g., rebalancing suggestions, hedging strategies, etc.)
        3. Consider both immediate tactical actions and longer-term strategic adjustments
        4. Be compliant with US regulations and fiduciary responsibilities
        5. Include monitoring suggestions for ongoing risk management
        
        Provide your recommendations in JSON format with these keys: immediate_actions, strategic_adjustments, hedging_strategies, monitoring_plan, alerts, summary.
        """
        
        recommendations_response = llm.invoke(recommendations_prompt)
        
        # Extract the JSON from the response
        try:
            # Find JSON content between triple backticks if present
            if "```json" in recommendations_response:
                json_str = recommendations_response.split("```json")[1].split("```")[0].strip()
            elif "```" in recommendations_response:
                json_str = recommendations_response.split("```")[1].strip()
            else:
                json_str = recommendations_response.strip()
                
            recommendations = json.loads(json_str)
        except:
            # If JSON parsing fails, create a structured response
            recommendations = {
                "immediate_actions": ["Consult a financial advisor"],
                "strategic_adjustments": ["Unable to generate strategic recommendations"],
                "hedging_strategies": ["Unable to suggest hedging strategies"],
                "monitoring_plan": ["Regular portfolio review"],
                "alerts": ["System error in generating alerts"],
                "summary": "There was an error generating recommendations. Please consult a financial advisor for personalized advice."
            }
        
        state["recommendations"] = recommendations
        return state
    
    except Exception as e:
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Recommendations error: {str(e)}")
        state["recommendations"] = {"error": str(e)}
        return state

def generate_final_report(state: AgentState) -> AgentState:
    """Generate the final comprehensive report."""
    try:
        portfolio = state["portfolio_data"]
        market = state["market_analysis"]
        macro = state["macro_analysis"]
        sector = state["sector_analysis"]
        stock = state["stock_analysis"]
        risk = state["risk_assessment"]
        recommendations = state["recommendations"]
        errors = state.get("errors", [])
        
        # Prepare final report using LLM
        report_prompt = f"""
        You are a professional risk management advisor preparing a comprehensive report for a client. Based on all the analyses performed, create a clear, concise executive summary of the portfolio risk assessment.
        
        PORTFOLIO OVERVIEW:
        Total Value: ${portfolio.get('total_value', 0):,.2f}
        Number of Holdings: {len(portfolio.get('tickers', []))}
        
        RISK ASSESSMENT SUMMARY:
        Overall Risk Level: {risk.get('overall_risk_level', 'unknown')}
        
        KEY RECOMMENDATIONS SUMMARY:
        {recommendations.get('summary', 'No recommendations available')}
        
        Your executive summary should:
        1. Highlight the most critical findings across all analyses
        2. Clearly communicate the overall risk level and justification
        3. Emphasize key recommendations and their urgency
        4. Be written in professional but accessible language
        5. Include any critical alerts that require immediate attention
        
        Provide your report in JSON format with these keys: title, executive_summary, risk_highlights, critical_alerts, key_recommendations, disclaimer.
        """
        
        report_response = llm.invoke(report_prompt)
        
        # Extract the JSON from the response
        try:
            # Find JSON content between triple backticks if present
            if "```json" in report_response:
                json_str = report_response.split("```json")[1].split("```")[0].strip()
            elif "```" in report_response:
                json_str = report_response.split("```")[1].strip()
            else:
                json_str = report_response.strip()
                
            final_report = json.loads(json_str)
        except:
            # If JSON parsing fails, create a structured response
            final_report = {
                "title": "Portfolio Risk Assessment Report",
                "executive_summary": "This automated risk assessment has encountered errors. Please review the detailed sections and consult a financial advisor.",
                "risk_highlights": ["Error generating risk highlights"],
                "critical_alerts": ["System encountered errors in analysis"],
                "key_recommendations": ["Consult a financial advisor"],
                "disclaimer": "This report is for informational purposes only and does not constitute financial advice. Please consult a qualified financial advisor before making investment decisions."
            }
        
        # Add full analyses
        final_report["full_analyses"] = {
            "market_analysis": market,
            "macro_analysis": macro,
            "sector_analysis": sector,
            "stock_analysis": stock,
            "risk_assessment": risk,
            "recommendations": recommendations
        }
        
        # Add error log if there were errors
        if errors:
            final_report["errors"] = errors
        
        state["final_report"] = final_report
        return state
    
    except Exception as e:
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Report generation error: {str(e)}")
        state["final_report"] = {
            "title": "Portfolio Risk Assessment Report - Error",
            "executive_summary": f"Error generating report: {str(e)}",
            "disclaimer": "This report encountered errors. Please consult a financial advisor."
        }
        return state

# Define the workflow
def create_workflow():
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("market_analysis", market_analysis_agent)
    workflow.add_node("macro_analysis", macro_analysis_agent)
    workflow.add_node("sector_analysis", sector_analysis_agent)
    workflow.add_node("stock_analysis", stock_analysis_agent)
    workflow.add_node("risk_assessment", risk_assessment_agent)
    workflow.add_node("recommendations", recommendations_agent)
    workflow.add_node("final_report", generate_final_report)
    
    # Define edges
    # First, run analyses in parallel
    workflow.add_edge("market_analysis", "risk_assessment")
    workflow.add_edge("macro_analysis", "risk_assessment")
    workflow.add_edge("sector_analysis", "risk_assessment")
    workflow.add_edge("stock_analysis", "risk_assessment")
    
    # Then, generate recommendations based on risk assessment
    workflow.add_edge("risk_assessment", "recommendations")
    
    # Finally, generate the final report
    workflow.add_edge("recommendations", "final_report")
    workflow.add_edge("final_report", END)
    
    # Set the entry point - uses conditional branching
    workflow.set_entry_point("market_analysis")
    
    # Create parallel branches for other analysis agents
    workflow.add_edge(
        "market_analysis", 
        conditional_edge(
            lambda x: True,
            "macro_analysis",
            "macro_analysis"
        ),
        override=True
    )
    
    workflow.add_edge(
        "market_analysis", 
        conditional_edge(
            lambda x: True,
            "sector_analysis",
            "sector_analysis"
        ),
        override=True
    )
    
    workflow.add_edge(
        "market_analysis", 
        conditional_edge(
            lambda x: True,
            "stock_analysis",
            "stock_analysis"
        ),
        override=True
    )
    
    return workflow.compile()

# Conditional edge helper
def conditional_edge(condition_fn, if_true, if_false):
    def _conditional_edge(state):
        if condition_fn(state):
            return if_true
        return if_false
    return _conditional_edge

# Streamlit UI
def create_streamlit_app():
    st.set_page_config(
        page_title="Portfolio Risk Assessment System",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("Portfolio Risk Assessment System")
    st.markdown("""
    This system analyzes your investment portfolio to identify risks and provide management strategies.
    Upload your portfolio file (CSV or Excel) to get started.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Portfolio (CSV or Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        # Parse portfolio
        with st.spinner("Parsing portfolio..."):
            portfolio, error = parse_portfolio(uploaded_file)
            
        if error:
            st.error(f"Error parsing portfolio: {error}")
        elif portfolio:
            st.success("Portfolio parsed successfully!")
            
            # Show portfolio overview
            st.subheader("Portfolio Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Value", f"${portfolio['total_value']:,.2f}")
            with col2:
                st.metric("Number of Holdings", len(portfolio['tickers']))
            with col3:
                st.metric("Unique Sectors", len(set(h.get('sector', 'Unknown') for h in portfolio.get('holdings', []))))
            
            # Display portfolio holdings
            if 'raw_df' in portfolio:
                st.dataframe(portfolio['raw_df'])
            
            # Run analysis button
            if st.button("Run Risk Assessment"):
                # Create the workflow
                workflow = create_workflow()
                
                # Initialize state
                initial_state = {
                    "portfolio_data": portfolio,
                    "market_analysis": {},
                    "macro_analysis": {},
                    "sector_analysis": {},
                    "stock_analysis": {},
                    "risk_assessment": {},
                    "recommendations": {},
                    "errors": [],
                    "final_report": {}
                }
                
                # Execute the workflow
                with st.spinner("Running comprehensive risk assessment... This may take a few minutes."):
                    try:
                        # Set up progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Run analysis steps
                        status_text.text("Analyzing market conditions...")
                        progress_bar.progress(10)
                        
                        final_state = workflow.invoke(initial_state)
                        
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Analysis complete!")
                        time.sleep(1)
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Check for errors
                        if final_state.get("errors"):
                            st.warning("Analysis completed with some issues:")
                            for error in final_state.get("errors"):
                                st.warning(error)
                        
                        # Display results
                        display_results(final_state)
                        
                    except Exception as e:
                        st.error(f"Error running analysis: {str(e)}")
    
    # Sample data option
    st.markdown("---")
    if st.button("Use Sample Portfolio"):
        # Create sample portfolio
        sample_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JNJ', 'V', 'PG', 'JPM'],
            'shares': [100, 50, 20, 15, 30, 40, 60, 45, 50, 35]
        })
        
        # Save sample to a temporary file
        sample_file = "sample_portfolio.csv"
        sample_df.to_csv(sample_file, index=False)
        
        # Load the sample file
        with open(sample_file, 'rb') as f:
            st.session_state.sample_file = f.read()
        
        # Refresh the page to use the sample
        st.experimental_rerun()
    
    # If we have a sample file in session state, use it
    if hasattr(st.session_state, 'sample_file'):
        uploaded_file = st.session_state.sample_file
        
        # Parse portfolio
        with st.spinner("Parsing sample portfolio..."):
            portfolio, error = parse_portfolio(uploaded_file)
            
        if error:
            st.error(f"Error parsing portfolio: {error}")
        elif portfolio:
            st.success("Sample portfolio loaded successfully!")
            
            # Show portfolio overview
            st.subheader("Portfolio Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Value", f"${portfolio['total_value']:,.2f}")
            with col2:
                st.metric("Number of Holdings", len(portfolio['tickers']))
            with col3:
                st.metric("Unique Sectors", len(set(h.get('sector', 'Unknown') for h in portfolio.get('holdings', []))))
            
            # Display portfolio holdings
            if 'raw_df' in portfolio:
                st.dataframe(portfolio['raw_df'])
            
            # Run analysis button
            if st.button("Run Risk Assessment on Sample"):
                # Create the workflow
                workflow = create_workflow()
                
                # Initialize state
                initial_state = {
                    "portfolio_data": portfolio,
                    "market_analysis": {},
                    "macro_analysis": {},
                    "sector_analysis": {},
                    "stock_analysis": {},
                    "risk_assessment": {},
                    "recommendations": {},
                    "errors": [],
                    "final_report": {}
                }
                
                # Execute the workflow
                with st.spinner("Running comprehensive risk assessment... This may take a few minutes."):
                    try:
                        # Set up progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Run analysis steps
                        status_text.text("Analyzing market conditions...")
                        progress_bar.progress(10)
                        
                        final_state = workflow.invoke(initial_state)
                        
                        # Update progress
                        progress_bar.progress(100)
                        status_text.text("Analysis complete!")
                        time.sleep(1)
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Check for errors
                        if final_state.get("errors"):
                            st.warning("Analysis completed with some issues:")
                            for error in final_state.get("errors"):
                                st.warning(error)
                        
                        # Display results
                        display_results(final_state)
                        
                    except Exception as e:
                        st.error(f"Error running analysis: {str(e)}")

def display_results(state):
    """Display the analysis results in the Streamlit UI."""
    # Get the report
    report = state.get("final_report", {})
    
    # Display the executive summary
    st.header(report.get("title", "Portfolio Risk Assessment Report"))
    
    # Risk level indicator
    risk_level = report.get("full_analyses", {}).get("risk_assessment", {}).get("overall_risk_level", "unknown")
    risk_colors = {
        "low": "green",
        "moderate": "orange",
        "high": "red",
        "severe": "darkred",
        "unknown": "gray"
    }
    risk_color = risk_colors.get(risk_level.lower(), "gray")
    
    st.markdown(f"""
    <div style="background-color: {risk_color}; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">Overall Risk Level: {risk_level.upper()}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Executive summary
    st.subheader("Executive Summary")
    st.write(report.get("executive_summary", "No summary available."))
    
    # Critical alerts
    critical_alerts = report.get("critical_alerts", [])
    if critical_alerts and critical_alerts[0] != "Error generating risk highlights":
        st.subheader("⚠️ Critical Alerts")
        for alert in critical_alerts:
            st.warning(alert)
    
    # Key recommendations
    st.subheader("Key Recommendations")
    recommendations = report.get("key_recommendations", ["No recommendations available."])
    for i, rec in enumerate(recommendations):
        st.write(f"{i+1}. {rec}")
    
    # Risk highlights
    st.subheader("Risk Highlights")
    risk_highlights = report.get("risk_highlights", ["No risk highlights available."])
    for highlight in risk_highlights:
        st.write(f"• {highlight}")
    
    # Tabs for detailed analyses
    st.markdown("---")
    st.header("Detailed Analysis")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Market", "Macroeconomic", "Sector", "Stock-Specific", "Portfolio Metrics"])
    
    with tab1:
        market_analysis = state.get("market_analysis", {})
        st.subheader("Market Analysis")
        
        # Market sentiment
        st.write(f"**Market Sentiment:** {market_analysis.get('sentiment', 'Unknown')}")
        
        # Risk factors
        st.write("**Risk Factors:**")
        for factor in market_analysis.get("risk_factors", ["No factors identified."]):
            st.write(f"• {factor}")
        
        # Market data visualization
        market_data = market_analysis.get("raw_data", {})
        if market_data:
            st.subheader("Market Indices (3-Month Performance)")
            
            # Prepare data for chart
            indices = []
            changes = []
            
            for index, data in market_data.items():
                if index == '^GSPC':
                    index_name = 'S&P 500'
                elif index == '^DJI':
                    index_name = 'Dow Jones'
                elif index == '^IXIC':
                    index_name = 'Nasdaq'
                elif index == '^VIX':
                    index_name = 'VIX'
                else:
                    index_name = index
                
                indices.append(index_name)
                changes.append(data.get('3m_change', 0))
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(indices, changes)
            
            # Color bars based on value
            for i, bar in enumerate(bars):
                if changes[i] > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            ax.set_ylabel('3-Month Change (%)')
            ax.set_title('Market Indices Performance')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add data labels
            for i, v in enumerate(changes):
                ax.text(i, v + (1 if v >= 0 else -2), f"{v:.1f}%", 
                        ha='center', va='bottom' if v >= 0 else 'top')
            
            st.pyplot(fig)
    
    with tab2:
        macro_analysis = state.get("macro_analysis", {})
        st.subheader("Macroeconomic Analysis")
        
        # Key metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Inflation", macro_analysis.get("inflation", "Unknown"))
            st.metric("GDP", macro_analysis.get("gdp", "Unknown"))
        with col2:
            st.metric("Interest Rates", macro_analysis.get("interest_rates", "Unknown"))
            st.metric("Employment", macro_analysis.get("employment", "Unknown"))
        
        # Key factors
        st.write("**Key Macroeconomic Factors:**")
        for factor in macro_analysis.get("key_factors", ["No factors identified."]):
            st.write(f"• {factor}")
        
        # Summary
        st.write("**Summary:**")
        st.write(macro_analysis.get("summary", "No summary available."))
    
    with tab3:
        sector_analysis = state.get("sector_analysis", {})
        st.subheader("Sector Analysis")
        
        # Get sector allocation data
        sector_data = sector_analysis.get("raw_data", {}).get("sector_allocations", {})
        
        if sector_data:
            # Create pie chart of sector allocations
            fig, ax = plt.subplots(figsize=(10, 6))
            wedges, texts, autotexts = ax.pie(
                sector_data.values(),
                labels=sector_data.keys(),
                autopct='%1.1f%%',
                startangle=90
            )
            ax.set_title('Portfolio Sector Allocation')
            st.pyplot(fig)
        
        # Sector-specific risks
        st.write("**Sector-Specific Risks:**")
        for risk in sector_analysis.get("specific_risks", ["No sector-specific risks identified."]):
            st.write(f"• {risk}")
        
        # Rebalancing recommendations
        st.write("**Sector Rebalancing Recommendations:**")
        for rec in sector_analysis.get("rebalancing_recommendations", ["No recommendations."]):
            st.write(f"• {rec}")
    
    with tab4:
        stock_analysis = state.get("stock_analysis", {})
        st.subheader("Stock-Specific Analysis")
        
        # Get stock data
        stock_data = stock_analysis.get("raw_data", {}).get("stock_data", {})
        
        if stock_data:
            # Create a table of top holdings
            st.write("**Top Holdings Performance:**")
            
            top_data = []
            for ticker, data in stock_data.items():
                if 'name' in data and 'portfolio_weight' in data and 'current_price' in data:
                    top_data.append({
                        'Ticker': ticker,
                        'Name': data['name'],
                        'Weight (%)': f"{data['portfolio_weight']*100:.2f}%",
                        'Current Price': f"${data['current_price']:.2f}",
                        '1-Month Change': f"{data['1m_change']:.2f}%" if '1m_change' in data else 'N/A',
                        '3-Month Change': f"{data['3m_change']:.2f}%" if '3m_change' in data else 'N/A',
                        'Volatility (30d)': f"{data['volatility_30d']:.2f}%" if 'volatility_30d' in data else 'N/A'
                    })
            
            if top_data:
                top_df = pd.DataFrame(top_data)
                st.dataframe(top_df.sort_values(by='Weight (%)', ascending=False))
        
        # Stock-specific issues
        st.write("**Concerning Fundamentals:**")
        for issue in stock_analysis.get("concerning_fundamentals", ["No specific issues identified."]):
            st.write(f"• {issue}")
        
        # Volatility concerns
        st.write("**Volatility Concerns:**")
        for concern in stock_analysis.get("volatility_concerns", ["No significant volatility concerns."]):
            st.write(f"• {concern}")
    
    with tab5:
        risk_assessment = state.get("risk_assessment", {})
        st.subheader("Portfolio Risk Metrics")
        
        # Risk metrics
        metrics = risk_assessment.get("raw_metrics", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Annualized Volatility", 
                     f"{metrics.get('annualized_volatility', 'N/A')}%" 
                     if isinstance(metrics.get('annualized_volatility'), (int, float)) 
                     else metrics.get('annualized_volatility', 'N/A'))
            
            st.metric("Value at Risk (95%)", 
                     f"${metrics.get('var_95', 'N/A'):,.2f}" 
                     if isinstance(metrics.get('var_95'), (int, float)) 
                     else metrics.get('var_95', 'N/A'))
        
        with col2:
            st.metric("Conditional VaR (95%)", 
                     f"${metrics.get('cvar_95', 'N/A'):,.2f}" 
                     if isinstance(metrics.get('cvar_95'), (int, float)) 
                     else metrics.get('cvar_95', 'N/A'))
            
            st.metric("Maximum Drawdown", 
                     f"{metrics.get('max_drawdown', 'N/A')}%" 
                     if isinstance(metrics.get('max_drawdown'), (int, float)) 
                     else metrics.get('max_drawdown', 'N/A'))
        
        # Correlation matrix
        correlation_matrix = metrics.get("correlation_matrix", {})
        if correlation_matrix:
            st.write("**Correlation Matrix (Top Holdings):**")
            
            # Convert to DataFrame for display
            tickers = list(correlation_matrix.keys())
            corr_data = []
            
            for ticker1 in tickers:
                row = {'Ticker': ticker1}
                for ticker2 in tickers:
                    if ticker2 in correlation_matrix.get(ticker1, {}):
                        row[ticker2] = correlation_matrix[ticker1][ticker2]
                corr_data.append(row)
            
            if corr_data:
                corr_df = pd.DataFrame(corr_data)
                corr_df = corr_df.set_index('Ticker')
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
                ax.set_title('Correlation Matrix of Top Holdings')
                st.pyplot(fig)
        
        # Diversification assessment
        st.write("**Diversification Assessment:**")
        st.write(risk_assessment.get("diversification_assessment", "No diversification assessment available."))
        
        # Scenario vulnerabilities
        st.write("**Scenario Vulnerabilities:**")
        for scenario in risk_assessment.get("scenario_vulnerabilities", ["No scenario analysis available."]):
            st.write(f"• {scenario}")
    
    # Detailed recommendations
    st.markdown("---")
    st.header("Detailed Recommendations")
    
    recommendations = state.get("recommendations", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Immediate Actions")
        for action in recommendations.get("immediate_actions", ["No immediate actions identified."]):
            st.write(f"• {action}")
        
        st.subheader("Hedging Strategies")
        for strategy in recommendations.get("hedging_strategies", ["No specific hedging strategies identified."]):
            st.write(f"• {strategy}")
    
    with col2:
        st.subheader("Strategic Adjustments")
        for adjustment in recommendations.get("strategic_adjustments", ["No strategic adjustments identified."]):
            st.write(f"• {adjustment}")
        
        st.subheader("Monitoring Plan")
        for item in recommendations.get("monitoring_plan", ["No specific monitoring plan."]):
            st.write(f"• {item}")
    
    # Disclaimer
    st.markdown("---")
    st.caption(report.get("disclaimer", "This report is for informational purposes only and does not constitute financial advice. Please consult a qualified financial advisor before making investment decisions."))

# Main function to run the app
if __name__ == "__main__":
    create_streamlit_app()