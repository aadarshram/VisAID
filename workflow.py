
import os
from dotenv import load_dotenv
from utils import capture_image, tts
from agents import scene_describer, companion, tools, search_recall_memories
from typing import TypedDict, List
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.embeddings import Embeddings
from langchain_core.messages import get_buffer_string, HumanMessage, AIMessage
import tiktoken
from langchain_core.runnables import RunnableConfig

# Setup environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Memory s
memory = MemorySaver()

class State(TypedDict):
    recall_memories: List[str]
    scene_description: str
    img: str
    messages: List[tuple]

def route_tools(state: State):
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        Literal["tools", "__end__"]: The next step in the graph.
    """
    print("Starting route tools")
    print(state)
    input("ok")

    msg = state["messages"][-1]
    if msg.tool_calls:
        return "tools"

    return END

def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        State: The updated state with loaded memories.
    """
    print("Starting load memories")
    convo_str = get_buffer_string(state["messages"])
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    recall_memories = search_recall_memories.invoke(convo_str, config)
    state["recall_memories"] = recall_memories
    print("Leaving load memories", state)
    return state

def scene_desc_node(state: State):
    scene_goal = state["messages"][-1].content # Currently assume last message is the goal
    state["scene_description"] = scene_describer(state["img"], scene_goal, openai_api_key)
    return state

def companion_node(state: State):
    print("Starting companion node")
    print(state)
    input("ok")
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    state["messages"] = [companion(state["messages"], openai_api_key, recall_str, state["scene_description"])]
    return state

def need_scene_desc(state: State):
    if state["img"]:
        return "scene_desc"
    else:
        return "companion"

def agent_workflow():
    workflow = StateGraph(State)
    workflow.add_node(load_memories)
    workflow.add_node("scene_desc", scene_desc_node)
    workflow.add_node("companion", companion_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "load_memories")
    workflow.add_conditional_edges(
        "load_memories",
        need_scene_desc,
        {
            "scene_desc": "scene_desc",
            "companion": "companion"
        }
    )
    workflow.add_edge("scene_desc", "companion")
    workflow.add_conditional_edges("companion", route_tools, ["tools", END])
    workflow.add_edge("tools", "companion")


    graph = workflow.compile(checkpointer = memory)

    # # Get flowchart
    # import io
    # from PIL import Image
    # img = io.BytesIO(graph.get_graph().draw_mermaid_png())
    # img = Image.open(img)
    # img.save("flowchart.png")
    # print("Flowchart saved as flowchart.png")
    return graph

 
if __name__ == "__main__":
    import time
    while True:
        user_goal = input("What do you want to do? ")
        image = input("Do you want to upload an image? (y/n) ")
        if image == "y":
            img = capture_image()
        else:
            img = None
        agent = agent_workflow()
        config = {"configurable": {"user_id": "1", "thread_id": "1"}}
        state = State(messages=[HumanMessage(content=user_goal)], img =  img, recall_memories=[], scene_description="")
        start = time.time() 
        response = agent.invoke(state, config = config)
        output = response["messages"]
        end = time.time()
        print("FINAL OUTPUT---------------------------------------------------------------------------")
        print(output[-1].content, f"Time taken: {end-start}")
        # start = time.time()
        # flag = tts(output)
        # end = time.time()
        # if flag:
        #     print("Text to speech conversion successful.", f"Time taken: {end-start}")
        # else:
        #     print("Text to speech conversion failed.")