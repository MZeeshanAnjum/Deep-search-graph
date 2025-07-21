import json
from typing_extensions import Literal
from datetime import datetime
from typing import Annotated, Literal, Optional, List, Dict

from langchain_core.messages import  SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from src.utils.utils import search_wrapper
from src.utils.configuration import Configuration
from src.utils.llm import get_llm
from edr.prompts.coordinator import COORDINATOR_PROMPT
from edr.prompts.planner import PLANNER_PROMPT
from edr.prompts.reporter import REPORTER_PROMPT
from edr.schema.deer_state import State, Step, Plan, StepType

from langgraph.types import Command, interrupt
import asyncio
from datetime import datetime
from src.utils.logger import get_custom_logger

logger= get_custom_logger("EDR")

llm = get_llm()

class EDR_DEER:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(State)
        # Add nodes
        builder.add_node("coordinator", self.coordinator_node)
        builder.add_node("background_investigator", self.background_investigation_node)
        builder.add_node("planner", self.planner_node)
        builder.add_node("researcher_team", self.researcher_team)
        builder.add_node("researcher", self.researcher)
        builder.add_node("reporter", self.reporter)
        # Add edges
        builder.add_edge(START, "coordinator")
        builder.add_edge("coordinator", "background_investigator")
        builder.add_edge("background_investigator", "planner")
        builder.add_edge("planner", "researcher_team")
        builder.add_conditional_edges(
            "researcher_team",
            self._continue_to_research_or_report,
            ["reporter", "researcher"],
        )
        builder.add_edge("reporter", END)
        return builder.compile()


    def _continue_to_research_or_report(self,state: State):
        current_plan = state.get("current_plan")

        # print(f"/n Current plan : {current_plan.steps}")
    
        if all(step.execution_res for step in current_plan.steps):
            return "reporter"
        for step in current_plan.steps:
            # print(step.execution_res)
            if not step.execution_res:
                break
        return "researcher"
        # if step.step_type and step.step_type == StepType.PROCESSING:
        #     return "coder"



    async def _execute_agent_step(self,
        state: State, tool
    ) -> Command[Literal["researcher_team"]]:
        """Helper function to execute a step using the specified agent."""
        current_plan = state.get("current_plan")
        observations = state.get("observations", [])
        sources = state.get("sources", [])

        # Find the first unexecuted step
        current_step = None
        completed_steps = []
        for step in current_plan.steps:
            if not step.execution_res:
                current_step = step
                break
            else:
                completed_steps.append(step)

        if not current_step:
            logger.warning("No unexecuted step found")
            return Command(goto="researcher_team")

        logger.info(f"Executing step: {current_step.title}")


        results=await tool(query=current_step.title)
        # print(results)
        current_sources = results.get("sources", [])
        combined_content = results.get("content")
        # Update the step with the execution result
        current_step.execution_res = combined_content
        logger.info(f"Step '{current_step.title}' execution completed by")

        return Command(
            update={"observations": observations + [combined_content], "sources": sources + current_sources},
            goto="researcher_team",
        )


    async def _setup_and_execute_agent_step(self,
        state: State,
        config: RunnableConfig,
        agent_type: str,
        tool:str,
        llm=llm
    ) -> Command[Literal["researcher_team"]]:
        """Helper function to set up an agent with appropriate tools and execute a step.

        This function handles the common logic for both researcher_node and coder_node:
        1. Configures MCP servers and tools based on agent type
        2. Creates an agent with the appropriate tools or uses the default agent
        3. Executes the agent on the current step

        Args:
            state: The current state
            config: The runnable config
            agent_type: The type of agent ("researcher" or "coder")
            default_tools: The default tools to add to the agent

        Returns:
            Command to update state and go to researcher_team
        """
        configurable = Configuration.from_runnable_config(config)

            # Use default tools if no MCP servers are configured
        
        return await self._execute_agent_step(state, tool)


    #NODES

    def coordinator_node(self,
        state: State, config: RunnableConfig, llm=llm
    ):
        """Coordinator node that communicate with customers."""
        logger.info("Coordinator talking.")
        now = datetime.now()
        formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        configurable = Configuration.from_runnable_config(config)
        prompt=prompt=COORDINATOR_PROMPT.format(CURRENT_TIME= formatted, user_current_question=state.get("user_current_question"))
        messages=[SystemMessage(content=prompt)]

        response =llm.invoke(messages)
        # logger.info(result)
        content= response.content
        # print(content)


        return{
                "locale": "en-US",
                "research_topic": content,
            }
        


    async def background_investigation_node(self,state: State, config: RunnableConfig):
        logger.info("background investigation node is running.")
        configurable = Configuration.from_runnable_config(config)
        query = state.get("research_topic")

        background_investigation_results = None
        results= await search_wrapper(query)
        # print(results.get("content"),"nothing")

        return {"background_investigation_results": results.get("content")}
        

    def planner_node(self,
        state: State, config: RunnableConfig, llm =llm
    ):
        llm=llm
        """Planner node that generate the full plan."""
        logger.info(" \n Planner generating full plan \n")
        configurable = Configuration.from_runnable_config(config)
        now = datetime.now()
        formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        llm=llm.with_structured_output(Plan)
        prompt=PLANNER_PROMPT.format(CURRENT_TIME= formatted, max_step_num= configurable.max_steps_of_planner, Background_information=state.get("background_investigation_results"), current_user_question=state.get("user_current_question"))
        messages=[SystemMessage(content=prompt)]
        response= llm.invoke(messages)


        # print(f"\n {response}")

        if response.has_enough_context == True:
            return Command(update={"current_plan":response}, goto="reporter")

        # elif response.has_enough_context == False:
        return {"current_plan":response}

    def reporter(self,state,llm=llm):
        logger.info(" i am in reporter node")

        observations = state.get("observations", [])
        # print(observations)
        content=""
        for obs in observations:
            content+= obs
        now = datetime.now()
        formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        prompt= REPORTER_PROMPT.format(CURRENT_TIME=formatted, OBSERVATIONS= content)
        messages=SystemMessage(content=prompt)
        response=llm.invoke([messages])
        # print(response)

        return {"search_agent_response": response.content, "sources":state.get("sources") }
        

    def researcher_team(self, state):
        logger.info(" i am in researcher_team node")
        return state

    async def researcher(self,
        state: State, config: RunnableConfig
    ) -> Command[Literal["researcher_team"]]:
        """Researcher node that do research"""
        logger.info("Researcher node is researching.")
        configurable = Configuration.from_runnable_config(config)

        return await self._setup_and_execute_agent_step(
            state,
            config,
            "researcher",
            tool = search_wrapper
        )
    
    
    def visualize(self):
        # log.info("Getting visualized planner agent")
        png_image = self.graph.get_graph(xray=True).draw_mermaid_png()
        with open("graph_visualization.png", "wb") as f:
            f.write(png_image)

        logger.info("Graph saved as graph_visualization.png")

    async def run(self, user_question: str) -> dict:
        # final = None
        # async for event in self.graph.astream({"user_current_question": user_question}):
        #     final = event
        # return final
        return await self.graph.ainvoke({"user_current_question": user_question})


if __name__ == "__main__":

    request = {
        "search_query": "What is LangGraph and how does it compare to LangChain?"
    }


    async def main():
        graph=EDR_DEER()
        graph.visualize()

        # answer = await graph.run("What is LangGraph and how does it compare to LangChain?")
        # print(answer)
    

        # print("\n Final Answer:\n", json.dumps(final_answer, indent=2))

    asyncio.run(main())
    # visualize()
    
