import os
import requests
from typing import Dict, Any, List, Optional
from langsmith import traceable
from tavily import TavilyClient
from duckduckgo_search import DDGS
from datetime import datetime
import csv
import sqlite3
import time
import random
from googlesearch import search
from playwright.async_api import async_playwright
import time
import asyncio
from langchain_community.document_loaders import UnstructuredURLLoader
import random
from src.utils.logger import get_custom_logger
from src.utils.configuration import Configuration
import json
import ast

from typing import Dict, List, Union

from config import TAVILY_API_KEY

logger= get_custom_logger("search_tool")

def save_logs(input_token,output_token,model_name,search_query):
    #db connection
    conn = sqlite3.connect('results.db')
    c = conn.cursor()
    #create table
    c.execute("CREATE TABLE IF NOT EXISTS search_logs (input_token TEXT,output_token TEXT,model_name TEXT,search_query TEXT,timestamp TEXT)")
    c.execute("INSERT INTO search_logs (input_token,output_token,model_name,search_query,timestamp) VALUES (?,?,?,?,?)",
                (input_token,output_token,model_name,search_query,str(datetime.now())))
    conn.commit()

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=False):
    """
    Takes either a single search response or list of responses from search APIs and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.
    
    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
            
    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                # print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()

def format_sources(search_results):
    """Format search results into a bullet-point list of sources.
    
    Args:
        search_results (dict): Tavily search response containing results
        
    Returns:
        str: Formatted string with sources and their URLs
    """
    return [source["url"] for source in search_results["results"] if "url" in source]

class DuckDuckGoSearchException(Exception):
    pass

@traceable
def duckduckgo_search(query: str, max_results: int = 3, fetch_full_page: bool = False) -> Dict[str, List[Dict[str, str]]]:
    """Search the web using DuckDuckGo.
    
    Args:
        query (str): The search query to execute
        max_results (int): Maximum number of results to return
        
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Same as content since DDG doesn't provide full page content
    """
    try:
        with DDGS() as ddgs:
            results = []
            search_results = list(ddgs.text(query, max_results=max_results))
            
            for r in search_results:
                url = r.get('href')
                title = r.get('title')
                content = r.get('body')
                
                if not all([url, title, content]):
                    # print(f"Warning: Incomplete result from DuckDuckGo: {r}")
                    continue

                raw_content = content
                if fetch_full_page:
                    try:
                        # Try to fetch the full page content using curl
                        import urllib.request
                        from bs4 import BeautifulSoup

                        response = urllib.request.urlopen(url)
                        html = response.read()
                        soup = BeautifulSoup(html, 'html.parser')
                        raw_content = soup.get_text()
                        
                    except Exception as e:
                        logger.error(e)
                        # print(f"Warning: Failed to fetch full page content for {url}: {str(e)}")
                
                # Add result to list
                result = {
                    "title": title,
                    "url": url,
                    "content": content,
                    "raw_content": raw_content
                }
                results.append(result)
                time.sleep(random.uniform(0.5, 1.0))
            time.sleep(random.uniform(2.0, 4.0))
            return {"results": results}
    except Exception as e:
        raise DuckDuckGoSearchException(f"DuckDuckGo failed: {str(e)}") from e
    

@traceable
def tavily_search(query, include_raw_content=True, max_results=3, api_key=None):
    """ Search the web using the Tavily API.
    
    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily in the formatted string
        max_results (int): Maximum number of results to return
        
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available"""
     
    travily_key = TAVILY_API_KEY

    # print("travli api_key=================", api_key)
    # if not api_key:
        # raise ValueError("TAVILY_API_KEY environment variable is not set")
    tavily_client = TavilyClient(api_key=travily_key)
    return tavily_client.search(query, 
                         max_results=max_results, 
                         include_raw_content=include_raw_content)

@traceable
def perplexity_search(query: str, perplexity_search_loop_count: int) -> Dict[str, Any]:
    """Search the web using the Perplexity API.
    
    Args:
        query (str): The search query to execute
        perplexity_search_loop_count (int): The loop step for perplexity search (starts at 0)
  
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available
    """

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }
    
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "Search the web and provide factual information with sources."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()  # Raise exception for bad status codes
    
    # Parse the response
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Perplexity returns a list of citations for a single search result
    citations = data.get("citations", ["https://perplexity.ai"])
    
    # Return first citation with full content, others just as references
    results = [{
        "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source 1",
        "url": citations[0],
        "content": content,
        "raw_content": content
    }]
    
    # Add additional citations without duplicating content
    for i, citation in enumerate(citations[1:], start=2):
        results.append({
            "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source {i}",
            "url": citation,
            "content": "See above for full content",
            "raw_content": None
        })
    
    return {"results": results}


@traceable
def google_search(query, num_results):
    """
    Perform a Google search and return search results.

    Args:
        query (str): The search query.
        num_results (int): The number of results to retrieve.

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Same as content for Google results
    """
    results = []
    query = query if isinstance(query, str) else str(query)
    for i, result in enumerate(search(query, advanced=True, num_results=num_results)):
        results.append(
            {
                "url": result.url,
                "title": result.title,
                "content": result.description,
                "raw_content": result.description
            }
        )
        time.sleep(random.uniform(0.5, 1.0))
    # print(f"google_search  search_results\n: {search_results}")
    time.sleep(random.uniform(2.0, 4.0))
    return {"results": results}


def correct_json(text: str,search_query:str=None) -> str:
    """string to json and correct format using LLM"""
    
    
    messages = [
        {"role": "system", "content": "you will be provided text you task is to return for correct json format"},
        {"role": "user", "content": text}
    ]
    response = openrouter_completion(
        api_key=os.getenv("OPENROUTER_KEY"),
        model="gpt-4o-mini",
        messages=messages,
        search_query=search_query
    )
    return response['choices'][0]['message']['content']




from openai import OpenAI

import requests
from typing import Union, List, Dict, Optional, Any

def openrouter_completion(
    api_key: str, 
    model: str, 
    messages: Optional[List[Dict[str, str]]] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1248,
    response_format: Optional[Dict[str, str]] = None,
    search_query: str=None
) -> dict:
    """
    Get a completion from OpenRouter API.
    Supports both chat completions and regular completions.

    Args:
        api_key (str): The API key for authentication.
        model (str): The model to use for the completion.
        messages (List[Dict[str, str]], optional): List of message objects for chat completions.
        prompt (str, optional): The prompt to send for regular completions.
        temperature (float, optional): Controls randomness. Defaults to 0.0.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1024.
        response_format (Dict[str, str], optional): Format for the response (e.g., JSON).

    Returns:
        dict: The response from the OpenRouter API.
    """
    # Import requests inside the function to ensure it's available
    import requests

    # Determine if we're using chat completions or regular completions
    if messages is not None:
        print(model,"is running...")
        url = "https://openrouter.ai/api/v1/chat/completions"
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        input_token = messages[-1]['content']
        if response_format:
            data["response_format"] = response_format
    elif prompt is not None:
        url = "https://openrouter.ai/api/v1/completions"
        data = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        input_token = prompt
    else:
        raise ValueError("Either 'messages' or 'prompt' must be provided")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app-or-website.com",  # Required by OpenRouter
        "X-Title": "Research Assistant"  # Optional title for tracking
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # Raise an exception for HTTP errors
    save_logs(len(input_token),len(response.text),model,search_query)
    return response.json()


@traceable
async def search_bing(query:str):
    """ 
    Web search tool
     Args:
        query (str): The search query to be submitted to Bing """

    

    user_agent = random.choice([
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/37.0.2062.94 Chrome/37.0.2062.94 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/600.8.9 (KHTML, like Gecko) Version/8.0.8 Safari/600.8.9",
    "Mozilla/5.0 (iPad; CPU OS 8_4_1 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Version/8.0 Mobile/12H321 Safari/600.1.4",
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.10240",
    "Mozilla/5.0 (Windows NT 6.3; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0",
    "Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36"
])

    async def langchain_loader(real_url: str):
        # Fetch page content using LangChain
        loader = UnstructuredURLLoader(urls=[real_url])
        docs = loader.load()
        content = docs[0].page_content if docs else "No content available"
        return content

    async def search(query: str, num_results: int = Configuration.max_search_results) -> dict:

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,  # Run in visible mode now
                channel="chrome"
            )
            context = await browser.new_context(user_agent=user_agent)
            page = await context.new_page()

            await page.goto(Configuration.BING_SEARCH_URL)
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(random.uniform(0.3, 1))

            try:
                await page.click("button:has-text('Accept')", timeout=5000)
                await asyncio.sleep(random.uniform(0.3, 1))

            except:
                pass

            try:

                await page.wait_for_selector("input#sb_form_q, textarea#sb_form_q", timeout=10000)
                await asyncio.sleep(random.uniform(0.3, 1))

            except Exception as e:
                logger.error(f"Error waiting for search box: {e}", exc_info=True)
                await browser.close()
                raise

            try:
                search_box_selector = "input[name='q'], textarea[name='q']"
                await page.fill(search_box_selector, query)
                await asyncio.sleep(random.uniform(0.3, 1))
                await page.press(search_box_selector, "Enter")
                await asyncio.sleep(random.uniform(0.3, 1))
                await page.wait_for_selector("#b_results", timeout=10000)
                # await asyncio.sleep(3)
                await asyncio.sleep(random.uniform(0.3, 1))
                await page.mouse.wheel(0, random.randint(100, 400))

                search_results = []
                result_elements = page.locator("#b_results .b_algo")
                count = await result_elements.count()

                for i in range(count):
                    try:
                        title_element = result_elements.nth(i).locator("h2 a")
                        await asyncio.sleep(random.uniform(0.3, 1))
                        link = await title_element.get_attribute("href", timeout=5000)
                        if not link:
                            continue
                        try:
                            new_page = await context.new_page()
                            await new_page.goto(link, wait_until="domcontentloaded", timeout=10000)
                            await asyncio.sleep(random.uniform(0.3, 1))
                            # time.sleep(4)
                            # await asyncio.sleep(random.uniform(0.3, 0.6))
                            real_url = new_page.url
                            await new_page.close()
                        except Exception as e:
                            logger.error(f"Error occurred while opening new page: {e}", exc_info=True)
                            continue

                        get_content = await langchain_loader(real_url)

                        result = {
                            "web_link": real_url,
                            "page_content": get_content
                        }

                        if len(search_results) >= num_results:
                            break
                        
                        search_results.append(result)

                        # search_results.append(result)

                        logger.info(f"\nResult {len(search_results)}:")
                        logger.info(f"URL: {real_url}")
                        # logger.info(f"Preview: {get_content[:30]}...\n")

                        
                    except Exception as e:
                        logger.error(f"Error processing result {i+1}: {str(e)}")
                        continue

                await browser.close()

                if not search_results:
                    raise RuntimeError("Search failed: No results were found.")
                
                return search_results

            except Exception as e:
                logger.error(f"Error occurred while searching: {e}", exc_info=True)
                raise
            
    return await search(query=query)

def formate_google_search(raw_response: Dict) -> Dict[str, object]:
    sources: List[str] = []
    all_content: List[str] = []

    for result in raw_response.get("results", []):
        url = result.get("url", "").strip()
        content = result.get("content") or result.get("raw_content") or ""
        if url:
            sources.append(url)
        if content.strip():
            all_content.append(content.strip())

    return {
        "sources": sources,
        "content": "\n\n".join(all_content)
    }

def format_bing_search(results: List[Dict[str, Union[str, None]]]) -> Dict[str, Union[List[str], str]]:
    sources: List[str] = []
    content_parts: List[str] = []

    for item in results:
        # Try Bing format
        url = item.get("web_link") or item.get("url")
        content = item.get("page_content") or item.get("content") or item.get("raw_content")

        if url:
            sources.append(url.strip())
        if content and isinstance(content, str) and content.strip():
            content_parts.append(content.strip())

    return {
        "sources": sources,
        "content": "\n\n".join(content_parts)
    }

def format_tavily_response(response: dict) -> dict:
    results = response.get("results", [])
    answer = response.get("answer")

    sources = []
    content_parts = []

    for r in results:
        url = r.get("url")
        if url:
            sources.append(url)

        content = r.get("content")
        if content:
            try:
                parsed = ast.literal_eval(content)
                if isinstance(parsed, dict):
                    content_parts.append(json.dumps(parsed, indent=2))
                else:
                    content_parts.append(str(parsed))
            except (ValueError, SyntaxError):
                content_parts.append(content)

    combined_content = answer or "\n\n".join(content_parts)

    return {
        "content": combined_content,
        "sources": sources
    }



@traceable

async def search_wrapper(query:str):

    try:
        logger.info("In bing search tool")
        # raise Exception("Something went wrong")
        results= await search_bing(query=query)
        return format_bing_search(results)
    
    except Exception as e:
        logger.info("Bing search failed, moved to Google search")

        try:
            # raise Exception("Something went wrong")
            results = google_search(query, num_results= Configuration.max_search_results)
            return formate_google_search(results)
        
        except Exception as e:
            logger.info("Bing and Google search  failed, moved to DuckDuckGo search")

            try:

                results = duckduckgo_search(query, max_results= Configuration.max_search_results, fetch_full_page=Configuration.fetch_full_page)
                # print(results)
                return results
            except Exception as e:
                logger.error(e)
                try:
                    logger.info("Tavily search")
                    results = tavily_search(query, include_raw_content=True, max_results=Configuration.max_search_results)
                    return format_tavily_response(results)  # If formatting is needed
                except Exception as e:
                    logger.exception("All search engines failed: %s", str(e))
                    return {"web_link": [""], "page_content": ""}

            

                                
                    




        
    
        
        

