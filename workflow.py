
import os
from dotenv import load_dotenv
from utils import capture_image, tts
from agents import scene_describer, companion
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class State(TypedDict):
    user_goal: str
    scene_description: str
    img: str
    response: str

def scene_desc_node(state: State):
    state["scene_description"] = scene_describer(state["img"], state["user_goal"], openai_api_key)
    return state

def companion_node(state: State):
    state["response"] = companion(state["user_goal"], openai_api_key, state["scene_description"])
    return state

def need_scene_desc(state: State):
    if state["img"]:
        return "scene_desc"
    else:
        return "companion"

def agent_workflow():
    workflow = StateGraph(State)
    workflow.add_node("scene_desc", scene_desc_node)
    workflow.add_node("companion", companion_node)
    
    workflow.add_conditional_edges(
        START,
        need_scene_desc,
        {
            "scene_desc": "scene_desc",
            "companion": "companion"
        }
    )
    workflow.add_edge("scene_desc", "companion")
    workflow.add_edge("companion", END)

    graph = workflow.compile()
    return graph
    
if __name__ == "__main__":
    import time
    user_goal = input("What do you want to do? ")
    image = input("Do you want to upload an image? (y/n) ")
    if image == "y":
        img = capture_image()
    else:
        img = None
    agent = agent_workflow()
    state = {"user_goal": user_goal, "img": img, "scene_description": None, "response": None} 
    start = time.time() 
    response = agent.invoke(state)
    output = response["response"]
    end = time.time()
    print(output, f"Time taken: {end-start}")
    start = time.time()
    flag = tts(output)
    end = time.time()
    if flag:
        print("Text to speech conversion successful.", f"Time taken: {end-start}")
    else:
        print("Text to speech conversion failed.")