"""Configuration and client setup for the lyrics search application."""

import hashlib
import json
import os
from pathlib import Path

import redis
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisCache
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# API Keys from environment
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# Redis configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Initialize LangChain clients
llm_client = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY,
    temperature=0.0,
)

deepseek_client = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
)

# Initialize Tavily search tool with advanced settings
tavily_search = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=5,
    search_depth="advanced",
    include_raw_content=True,
)

# Initialize Redis cache
try:
    redis_client = redis.from_url(REDIS_URL)
    # Test connection
    redis_client.ping()

    # Set up LangChain Redis cache with proper configuration
    cache = RedisCache(redis_client)
    set_llm_cache(cache)

    print(f"✅ Redis cache initialized at {REDIS_URL}")
except Exception as e:
    print(f"⚠️ Redis cache not available: {e}")
    print("🔄 Running without cache")
    redis_client = None


# Search cache functions
def get_search_cache_key(query: str) -> str:
    """Generate a cache key for search queries."""
    return f"search:{hashlib.md5(query.encode()).hexdigest()}"


def get_cached_search(query: str):
    """Get cached search results."""
    if not redis_client:
        return None
    try:
        cache_key = get_search_cache_key(query)
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return None


def cache_search_results(query: str, results: dict, ttl: int = 3600):
    """Cache search results with TTL (default 1 hour)."""
    if not redis_client:
        return
    try:
        cache_key = get_search_cache_key(query)
        redis_client.setex(cache_key, ttl, json.dumps(results))
    except Exception:
        pass
