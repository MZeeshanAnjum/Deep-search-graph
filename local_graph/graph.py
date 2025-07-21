import time
import asyncio
import base64
from io import BytesIO
from typing import Annotated
from datetime import datetime
from pathlib import Path

from langgraph.graph import MessagesState
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.llm import get_llm
from local_search.prompts.prompts import GET_SEARCH_QUERY_PROMPT, SUMMARIZER_PROMPT, EVALUATION_PROMPT, \
    IMAGE_SUMMARIZER_PROMPT
from local_search.utils.search_tool import search_bing, image_search
from config import MAX_ITERATIONS, NO_OF_IMAGE_RESULTS_RETURNED, \
    MAX_SEARCH_RESULTS
from src.utils.logger import get_custom_logger
from src.utils.utils import search_wrapper

from local_search.schema.search_agent_schema import SearchAgentState, SearchQueryRequest, SearchAgentResponse


logger = get_custom_logger("search_agent_model")




class SearchAgent:
    def __init__(self):
        start_time = time.time()
        self.llm = get_llm()
        self.graph = self.get_search_graph()
        self.count_iterations = 1

    def get_search_queries_agent(self, state: SearchAgentState):
        search_assistant_result = []
        try:
            logger.info(f"[ Calling Query Generation Method... ]")

            system_prompt = SystemMessage(
                content=GET_SEARCH_QUERY_PROMPT.format(current_date=datetime.now().strftime("%Y-%m-%d"))
            )
            search_assistant_result = self.llm.with_structured_output(SearchQueryRequest).invoke(
                [system_prompt] + state["messages"])

            logger.info(f"search_queries => {search_assistant_result}")
        except Exception as e:
            logger.error(f"Error while getting search queries: {e}", exc_info=True)
        return {"search_queries": search_assistant_result}

    def process_image_search_queries(self, image_search_results: list[dict]):
        logger.info(f"[ Calling Image Query Processing Method... ]")
        image_search_text_label = []
        image_source_list = []
        image_url_list = []

        for result in image_search_results:
            image_label = result.get("text")
            source_in_bas64 = result.get("image_source")
            image_url = result.get("image_url")

            image_search_text_label.append(image_label)
            image_source_list.append(source_in_bas64)
            image_url_list.append(image_url)

        return image_search_text_label, image_source_list, image_url_list

    def process_web_search_result(self, search_results):
        logger.info(f"[ Calling Web Query Processing Method... ] ")
        web_search_text_label = []
        web_source_list = []


        # print(search_results)
        web_label = search_results.get("content")
        source = search_results.get("sources")
        logger.info(source)
        for web in (source):

            web_source_list.append(web)
        web_search_text_label.append(web_label)

        return web_search_text_label, web_source_list

    async def call_search_tool(self, state: SearchAgentState):
        logger.info(f"[ Calling Search Tool Method... ] ")
        web_search_result = []
        web_source_list = []

        ## check if image is passed with input query
        if state["image_path"]:  ## TODO:  should we check if query is empty?
            image_search_result = image_search(image_path=state["image_path"], num_results=NO_OF_IMAGE_RESULTS_RETURNED)
            ## call process image to get image_sources and text labels
            image_search_text_label, image_source_list, image_url_list = self.process_image_search_queries(
                image_search_result)
            return {"image_query_results": image_search_text_label, "image_sources": image_source_list,
                    "image_urls": image_url_list}

        else:
            ## check if recursion loop
            if self.count_iterations > 1:
                ## use targeted search query returned by evaluation agent
                search_query = state["evaluation_response"].content
                search_result = await search_wrapper(query=search_query)
                ## call process web to get web_sources and text labels
                web_search_result, web_source_list = self.process_web_search_result(search_result)

            else:
                ## normal execution
                for search_query in state["search_queries"].search_queries:
                    search_result = await search_wrapper(query=search_query)

                    # print(search_result)
                    ## call process web to get web_sources and text labels
                    web_search_result, web_source_list = self.process_web_search_result(search_result)
                    # print(web_seaarch_result)

            return {"query_results": web_search_result, "sources": web_source_list}

    def image_summarizer_agent(self, state: SearchAgentState):
        logger.info(f"[ Calling Image Summarizer Agent... ] ")
        user_query = state['messages'][0].content

        ## convert tool result list to string
        search_result_string = " ".join(state["image_query_results"])
        system_prompt = SystemMessage(
            content=IMAGE_SUMMARIZER_PROMPT.format(text_labels=search_result_string)
        )

        summarized_result = self.llm.invoke([system_prompt] + [HumanMessage(content=user_query)])

        logger.info(f"summary_response => {summarized_result}")
        return {"summary_response": summarized_result}

    def summarizer_agent(self, state: SearchAgentState):
        logger.info(f"[ Calling Summarizer Agent.. ] ")
        user_query = state['messages'][0].content
        system_prompt = SystemMessage(
            content=SUMMARIZER_PROMPT.format(user_query=user_query)
        )
        search_result_string = " ".join(state["query_results"])  ## convert tool result list to string

        summarized_result = self.llm.invoke([system_prompt] + [HumanMessage(content=search_result_string)])

        logger.info(f"summary_response => {summarized_result}")
        return {"summary_response": summarized_result}

    def evaluation_agent(self, state: SearchAgentState):
        logger.info(f"[ Calling Evaluation Agent Method... ] ")
        user_query = state['messages'][0].content   
        system_prompt = SystemMessage(
            content=EVALUATION_PROMPT.format(user_query=user_query)
        )
        evaluation_response = self.llm.invoke(
            [system_prompt] + [HumanMessage(content=state["summary_response"].content)])

        logger.info(f"Evaluation Agent response: {evaluation_response}")

        ## append previous response to final response state
        return {"evaluation_response": evaluation_response, "final_response": state["summary_response"].content}

    def should_call_search_tool(self, state: SearchAgentState):
        ## check if search tool should be called or the query analyzer

        if state["image_path"]:
            return "search_tool_agent"
        else:
            return "search_query_agent"

    def get_final_summary(self, state: SearchAgentState):
        ## check which summarizer agent to call

        if state["image_path"] :
            return "image_summarizer_agent"
        else:
            return "summarizer_agent"

    def should_continue_next_iteration(self, state: SearchAgentState):
        ## check if the iteration should continue

        logger.info(f"Graph Iteration Count :  {self.count_iterations}")

        if state["evaluation_response"].content == "END" or self.count_iterations >= MAX_ITERATIONS:
            logger.info(f"Graph Execution Ended at Iteration {self.count_iterations}")
            return END
        else:
            ## update iteration counter
            self.count_iterations += 1
            return "search_tool_agent"

    def get_search_graph(self):
        build_search_graph = StateGraph(SearchAgentState)

        ## add nodes
        build_search_graph.add_node("search_query_agent", self.get_search_queries_agent)
        build_search_graph.add_node("search_tool_agent", self.call_search_tool)
        build_search_graph.add_node("summarizer_agent", self.summarizer_agent)
        build_search_graph.add_node("image_summarizer_agent", self.image_summarizer_agent)
        build_search_graph.add_node("evaluation_agent", self.evaluation_agent)

        ## add edges
        build_search_graph.add_conditional_edges(START, self.should_call_search_tool,
                                                 ["search_query_agent", "search_tool_agent"])
        build_search_graph.add_edge("search_query_agent", "search_tool_agent")
        build_search_graph.add_conditional_edges("search_tool_agent", self.get_final_summary,
                                                 ["summarizer_agent", "image_summarizer_agent"])
        build_search_graph.add_edge("image_summarizer_agent", "evaluation_agent")
        build_search_graph.add_edge("summarizer_agent", "evaluation_agent")

        ## add conditional edge for evaluation of response
        build_search_graph.add_conditional_edges("evaluation_agent", self.should_continue_next_iteration,
                                                 ["search_tool_agent", END])

        search_graph = build_search_graph.compile()

        return search_graph

    def visualize(self):
        logger.info("Getting visualized planner agent")

        # Get the PNG image as bytes
        png_image = self.graph.get_graph(xray=True).draw_mermaid_png()
        
        # Save image to disk
        
        output_path = Path("outputs/planner_agent_graph.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
        with open(output_path, "wb") as f:
            f.write(png_image)
        logger.info(f"Saved planner graph image to {output_path.resolve()}")

    async def get_search_agent_response(self, query: str, image_path=None):
        base64_image_results = None
        image_urls = None
        logger.info("[ Invoking Search Agent Graph... ]")
        search_agent_final_response = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=query)], "image_path": image_path}
        )
        
        search_agent_response = search_agent_final_response.get("final_response", None)
        base64_image_results = search_agent_final_response.get("image_sources", None)
        image_urls = search_agent_final_response.get("image_urls", None)
        sources = search_agent_final_response.get("sources", None)
        
        return search_agent_response, base64_image_results, image_urls, sources

    async def invoke_search_agent(self, query: str, image_path=None):
        search_agent_response, image_results, image_urls, sources = await self.get_search_agent_response(
            query=query,
            image_path=image_path
        )

        image_results = image_results or []
        image_urls = image_urls or []
        sources = sources or []
        return SearchAgentResponse(
            search_agent_response=search_agent_response,
            image_results=image_results,
            image_urls=image_urls,
            sources=sources
        )

if __name__== "__main__":
    agent= SearchAgent()
    agent.visualize()
