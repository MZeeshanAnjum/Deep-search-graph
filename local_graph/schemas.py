from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Annotated
from langgraph.graph import MessagesState


class SearchAgentState(MessagesState):
    image_path: str 
    summary_response: str
    evaluation_response: str

    search_queries: Annotated[list[str], "A list of search queries"]
    query_results: Annotated[list[str], "Web Results from search queries"]
    sources: Annotated[list[str], "Source Links of Web Search Queries"]
    image_query_results: Annotated[list[dict], "Image Results from search queries"]
    image_sources: Annotated[list[str], "base64 Source Links of Web Search For image"]
    image_urls: Annotated[list[str], "URL Links of Web Search For images"]
    final_response: Annotated[list[str], "Final evaluated response"]



class SearchQueryRequest(BaseModel):
    search_queries: List[str]

class SearchAgentResponse(BaseModel):
    search_agent_response: str
    image_results: list[str] =[]
    image_urls: List[str] =[]
    sources: List[str] = []
