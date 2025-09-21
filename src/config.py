"""Configuration and client setup for the lyrics search application."""

import os

import openai
from tavily import TavilyClient

# API Keys from environment
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize clients
llm_client = openai.OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
