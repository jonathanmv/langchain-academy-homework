from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

llm = ChatOpenAI(model="gpt-4o-mini")

# Namespace is a tuple
namespace_for_users = "users"
key_for_user_profile = "profile"

def read_user_profile(store:BaseStore, user_id: str):
    namespace = (namespace_for_users, user_id)
    profile = store.get(namespace, key_for_user_profile)
    if profile:
        return profile.value.get("profile")

    return None


def write_user_profile(store: BaseStore, user_id: str, profile: str):
    if not profile:
        raise ValueError("Profile cannot be empty")

    namespace = (namespace_for_users, user_id)
    return store.put(namespace, key_for_user_profile, { "profile": profile })

class ConversationState(MessagesState):
    user_id: str
    profile: str | None

def load_user_profile(state: ConversationState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    profile = read_user_profile(store, user_id)
    return {"user_id": user_id, "profile": profile}

MODEL_SYSTEM_MESSAGE = """
You are a helpful assistant with memory that provides information about the user.
If you have memory about this user, use it to personalize your responses.
Here's the user's profile (could be empty): {profile}
"""

def call_model(state: ConversationState):
    system_message = MODEL_SYSTEM_MESSAGE.format(profile=state["profile"])
    messages = [
        SystemMessage(content=system_message),
        *state["messages"]
    ]

    response = llm.invoke(messages)
    return {"messages": response}


CREATE_USER_PROFILE_SYSTEM_MESSAGE = """You are collecting information about a user to personalize your responses.
Current user profile: {profile}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
    - Personal details (name, age, location, etc.)
    - User preferences (favorite topics, hobbies, etc.)
    - User goals (what they want to achieve, what they're struggling with, etc.)
    - User interests (what they like, what they don't like, etc.)
3. Merge any new information with the existing profile
4. Format the profile as a clear, bulleted list
5. If new information conflicts with the existing profile, keep the most recent version

Remember: Only include factual information directly stated by the user. Do not make up information.

Based on the chat history below, please update the user profile:
"""

def update_user_profile(state: ConversationState, config: RunnableConfig, store: BaseStore):
    user_id = state["user_id"]
    old_profile = state["profile"]
    messages = state["messages"]

    system_message = CREATE_USER_PROFILE_SYSTEM_MESSAGE.format(profile=old_profile)
    messages = [
        SystemMessage(content=system_message),
        *messages
    ]

    response = llm.invoke(messages)
    new_profile = response.content

    write_user_profile(store, user_id, new_profile)
    return {"profile": new_profile}


builder = StateGraph(ConversationState)
builder.add_node(load_user_profile)
builder.add_node(call_model)
builder.add_node(update_user_profile)

builder.add_edge(START, "load_user_profile")
builder.add_edge("load_user_profile", "call_model")
builder.add_edge("call_model", "update_user_profile")
builder.add_edge("update_user_profile", END)



across_threads_memory = InMemoryStore()
within_thread_memory = MemorySaver()
graph = builder.compile(checkpointer=within_thread_memory, store=across_threads_memory)

# print(graph.get_graph().draw_ascii())
#      +-----------+
#      | __start__ |
#      +-----------+
#             *
#             *
#             *
#  +-------------------+
#  | load_user_profile |
#  +-------------------+
#             *
#             *
#             *
#     +------------+
#     | call_model |
#     +------------+
#             *
#             *
#             *
# +---------------------+
# | update_user_profile |
# +---------------------+
#             *
#             *
#             *
#       +---------+
#       | __end__ |
#       +---------+

user_id = input("Enter user ID: ")
thread_id = input("Enter thread ID: ")

while True:
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
    message = input("Enter message (or quit, new, or profile): ")

    if message == "quit":
        break

    if message == "profile":
        print(graph.get_state(config).values.get("profile"))
        continue

    if message == "new":
        user_id = input("Enter user ID: ")
        thread_id = input("Enter thread ID: ")
        continue

    for event in graph.stream({"messages": [HumanMessage(content=message)]}, config=config, stream_mode="values"):
        event["messages"][-1].pretty_print()

    print("---"*33)

