# Agent functions

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI 
from langchain_core.documents import Document
import uuid
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import os
import faiss
from typing import List
from dotenv import load_dotenv
# Tools for memory


# Setup environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API Key not found")

# For context memory
VECTORSTORE_FILE = "vectorstore.faiss"  # File for local FAISS persistence

# Load existing FAISS vector store or create a new one
try:
    recall_vectorstore = FAISS.load_local(VECTORSTORE_FILE, OpenAIEmbeddings(api_key=openai_api_key), allow_dangerous_deserialization=True)
    print("Loaded existing vector store.")
except:
    index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query("hello world")))

    recall_vectorstore = FAISS(
        embedding_function=OpenAIEmbeddings(),
        index=index,
        docstore= InMemoryDocstore(),
        index_to_docstore_id={}
    )   
    recall_vectorstore.save_local(VECTORSTORE_FILE)
    print("Initialized a new vector store.")

# Tools to search and store memory

def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save memory")
    return user_id

@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save conversation memory to vectorstore for later semantic retrieval to aid in providing context-aware responses."""
    user_id = get_user_id(config)
    document = Document(
        page_content = memory, id = str(uuid.uuid4()), metadata = {"user_id": user_id}
    )
    recall_vectorstore.add_documents([document])
    print(document)
    return f"Memory saved for long-term: {memory}"

@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories to add context and provide personalized responses."""
    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        print(type(doc))
        print(doc)
        return doc.get("user_id") == user_id

    documents = recall_vectorstore.similarity_search(
        query, k=3, filter=_filter_function
    )
    print(documents)
    return [f"Memory found: {document.page_content}" for document in documents]

tools = [save_recall_memory, search_recall_memories]





# Scene describer: Describe the scene based on an image and user goal

def scene_describer(image_base64, user_goal, openai_api_key):
    image_data_url = f"data:image/jpeg;base64,{image_base64}"
    messages = [
    SystemMessage(
        content=[
            {"type": "text", "text": """
                You are a scene description assistant for a visual aid companion to help a blind user. Your goal is to provide detailed, context-aware descriptions of the users surroundings based on images. The description should prioritize objects and landmarks that are relevant to the user's request. Your task is to describe the scene clearly, providing enough information to the visual aid companion to help user navigate and interact with their environment.

                Instructions:
                Focus on Relevant Objects: Your description should center around the objects that are most relevant to the user's request. This could be anything from furniture, objects of interest, obstacles, or landmarks. The goal is to provide information to the visual aid companion to help the user understand their surroundings to accomplish a specific task (e.g., navigating a room, finding an object, etc.).
                Spatial Details: Include details on the location, orientation, and distances between objects. Mention relative positions (e.g., “to the left of the user,” “ahead of the user,” “on the right side of the table”) and how they are laid out in space.
                Important Objects & Landmarks: Identify any key items or landmarks that might help the user navigate (e.g., “a table with a mug,” “a door in front of the user”, “a chair to your right”).
                Actionable Descriptions: The description should provide actionable data to the visual aid companion to form instructions for the user.
                No Assumptions or Specific Tasks: Do not assume a specific task. The user may request a wide variety of tasks, and your description should be flexible enough to handle any situation. For instance, if the user asks to find an object or navigate to a location, base your description on what is present in the scene.
                Clarity and Precision: Ensure your description is precise and clear. Avoid ambiguous terms unless necessary. Use simple language that can easily be understood by the visual aid companion and the user.
                Tone: The tone should be neutral, helpful, and informative. You are providing the visual aid companion information to understand the user environment, not making assumptions or offering opinions.

                Example 1 (General Task):
                User Request: "Can you tell me what is in front of me?"
                Image: [User uploads a picture of a room]
                Desired Output: "In front of the user, there is a table with a mug placed near the left corner. To the right, there is a chair. The room seems to be well-lit with a large window behind the table."
                Example 2 (General Task):
                User Request: "Can you help me find the nearest door?"
                Image: [User uploads a picture of a hallway]
                Desired Output: "There is a door about 3 meters in front of the user. It is on the left side of the hallway, with a small rug in front of it."

                Prompt:
                Analyze the image provided and describe the scene with the focus on relevant objects, spatial details, important landmarks, and actionable descriptions. Your goal is to provide scene description to the visual aid companion to help user understand their surroundings and navigate effectively. Avoid assumptions and be clear and precise in your description.
                """
        }
        ]
    ),
    HumanMessage(
        content=[
        {"type": "text", "text": user_goal},
        {
            "type": "image_url",
            "image_url": {"url": image_data_url},
        },
    ]
    )
]
    model = ChatOpenAI(model = "gpt-4o-mini", api_key = openai_api_key)
    response = model.invoke(messages)
    return response.content

# Companion: Assist the user in achieving their goal based on the scene description if provided

def companion(user_goal, openai_api_key, recall_memories=None, scene_description=None):
    messages = [
        SystemMessage(
            content="""
You are an AI companion designed to assist a blind user. Your primary goal is to provide clear, supportive, and actionable guidance to help the user understand their surroundings and accomplish tasks.

### Key Capabilities:
1. **Natural Conversation:**  
   Respond empathetically and adapt to the user’s emotional and situational context.
   
2. **Scene Understanding:**  
   Use scene descriptions (if provided) to give context-aware advice. Encourage the user to share images for better assistance.
   
3. **Actionable Advice:**  
   Offer clear, step-by-step instructions for tasks like locating objects or navigating spaces.

4. **Memory Utilization:**  
   - Save important user information for long-term use with the memory tool (`save_recall_memory`).
   - Search saved memories (`search_recall_memories`) to provide personalized support.
   - Use memory to tailor responses and ensure consistency over time.

### Guidelines:
- Always ask for clarification if the request is unclear.
- Leverage memory and scene descriptions to provide detailed, accurate responses.
- Use an engaging and empathetic tone to create a natural, helpful interaction.

### Example Interaction:
- **User Request:** "Can you help me find my keys?"  
  - Response: "Try checking nearby surfaces. If you share an image, I can assist further."
- **User Info:** "I keep my keys in the kitchen drawer."  
  - *Tool:* [Memory saved]  
  - Response: "Got it! I’ll remember you keep your keys in the kitchen drawer."
"""
        )
    ]

    if scene_description:
        messages.append(
            SystemMessage(
                content=f"Scene analysis: {scene_description}."
            )
        )
    messages.append(
        SystemMessage(content=f"Recall memory: {recall_memories or 'No relevant memories.'}")
    )
    messages.append(
        HumanMessage(content=f"User goal: {user_goal}")
    )

    model = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    model_with_tools = model.bind_tools(tools)

    return model_with_tools.invoke(messages)

if __name__ == "__main__":


    # # Test scene description
    # user_goal = "Can you help me find my coffee?"
    # from utils import capture_image
    # img = capture_image()
    
    # response = scene_describer(img, user_goal, openai_api_key)
    # print(response.content)

    # Test companion
    user_goal = "User goal: Find my coffee"
    scene_desc = "Your coffee is located on the table to your right. It is in a green cup. On the table, there is also a red item, possibly a notebook or tablet, near the center, and a gray water bottle further back. There are some snacks in a container to the right of the coffee. The table surface appears to be wooden and has a few items scattered across it."
    response = companion(user_goal, openai_api_key, None, scene_description = None)
    print(response.content)

