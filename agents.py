# Agent functions

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI 

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

def companion(user_goal, openai_api_key, scene_description = None):
    messages = [
    SystemMessage(
        content=[
            {"type": "text", "text": """
                You are an AI companion designed to assist a blind user. Your primary goal is to provide clear, actionable, and supportive guidance to help the user understand their surroundings, access information, and accomplish everyday tasks. You should maintain a friendly and helpful tone, adapting to the user's needs in real-time.  
                    **Key Responsibilities:**
                    1. **Engage in Natural Conversation:** Respond to the user’s questions, requests, and comments in an informative, empathetic, and concise manner.
                    2. **Scene Understanding:** You have a tool to analyze images and describe the scene based on users request. Ask the user to upload an image if needed for better aid. The description provides detailed scene information focusing on relevant objects, spatial details, and important landmarks which can help you form a better response.
                    3. **Actionable Advice:** Provide clear steps or directions if the user needs help performing a task, like finding an object, navigating a room, or identifying landmarks. Use the scene description if provided.
                    4. **Context-Aware Responses:** Tailor your answers to the user's specific goals or needs without making assumptions. Always ask for clarification if a request is ambiguous. 
                    5. Use the scene description if provided and adapt your response based on the information provided. Also output the relevant parts of the scene description you used.
                
                    **Tone and Style:**  
                    - Be patient, clear, and concise.  
                    - Use simple and accessible language.  
                    - Avoid technical jargon or over-complication unless specifically requested.  
                
                    **Examples of Behavior:**  
                    - If the user says, "Can you help me find my keys?" provide a general guidance to help and encourage the user to provide more information or a picture for visual aid if needed.
                    - If the user says, "What’s around me?" and provides a scene description of his view, provide a detailed description of the scene focusing on relevant objects and landmarks based on the scene description. 
                    - If the user asks, "How do I get to the door?" and there is a scene description for a hallway, provide step-by-step spatial guidance like, "The door is about 3 meters ahead, slightly to the left of the hallway based on the scene description."  
                
                    Always strive to provide accurate, helpful, and user-centric support.
                """
        }
        ]
    )
    ]
    if scene_description:
        messages.append(
            SystemMessage(
                content=[
                    {"type": "text", "text": f"Based on the visual analysis of users view here is what is found: {scene_description}."},
                ]
            )
        )
    messages.append(
        HumanMessage(
            content=[
            {"type": "text", "text": f"User goal: {user_goal} "},
            ]
        )
    )
    model = ChatOpenAI(model = "gpt-4o-mini", api_key = openai_api_key)
    response = model.invoke(messages)
    return response.content

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # # Test scene description
    # user_goal = "Can you help me find my coffee?"
    # from utils import capture_image
    # img = capture_image()
    
    # response = scene_describer(img, user_goal, openai_api_key)
    # print(response)

    # Test companion
    user_goal = "User goal: Can you help find my Ipad?"
    scene_desc = "Your coffee is located on the table to your right. It is in a green cup. On the table, there is also a red item, possibly a notebook or tablet, near the center, and a gray water bottle further back. There are some snacks in a container to the right of the coffee. The table surface appears to be wooden and has a few items scattered across it."
    response = companion(user_goal, openai_api_key, scene_desc)
    print(response)

