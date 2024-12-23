from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from typing import TypedDict, List, Optional
from pydantic import BaseModel, ValidationError, Field
from trustcall import create_extractor

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

class UserProfile(TypedDict):
    user_name: str
    interests: List[str]

def read_user_profile(store:BaseStore, user_id: str):
    namespace = (namespace_for_users, user_id)
    profile = store.get(namespace, key_for_user_profile)
    if profile:
        return profile.value

    return None


def write_user_profile(store: BaseStore, user_id: str, profile: UserProfile):
    if not profile:
        raise ValueError("Profile cannot be empty")

    namespace = (namespace_for_users, user_id)
    return store.put(namespace, key_for_user_profile, profile)

def format_profile(profile: UserProfile | None):
    if not profile:
        return "<no profile>"

    user_name = profile["user_name"]
    interests = ", ".join(profile["interests"])
    return f"User name: {user_name}\nInterests: {interests}"


MODEL_SYSTEM_MESSAGE = """
You are a helpful assistant with memory that provides information about the user.
If you have memory about this user, use it to personalize your responses.
Here's the user's profile (could be empty): {profile}
"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    profile = read_user_profile(store, config["configurable"]["user_id"])
    system_message = MODEL_SYSTEM_MESSAGE.format(profile=format_profile(profile))
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

def update_user_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    old_profile = format_profile(read_user_profile(store, user_id))
    messages = state["messages"]

    system_message = CREATE_USER_PROFILE_SYSTEM_MESSAGE.format(profile=old_profile)
    messages = [
        SystemMessage(content=system_message),
        *messages
    ]

    llm_with_structure = llm.with_structured_output(UserProfile)
    response = llm_with_structure.invoke(messages)

    write_user_profile(store, user_id, response)


builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_node(update_user_profile)

builder.add_edge(START, "call_model")
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

# user_id = input("Enter user ID: ")
# thread_id = input("Enter thread ID: ")

# while True:
#     config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
#     message = input("Enter message (or quit, new, or profile): ")

#     if message == "quit":
#         break

#     if message == "profile":
#         profile = read_user_profile(across_threads_memory, user_id)
#         print(f"profile: {format_profile(profile)}")
#         continue

#     if message == "new":
#         user_id = input("Enter user ID: ")
#         thread_id = input("Enter thread ID: ")
#         continue

#     for event in graph.stream({"messages": [HumanMessage(content=message)]}, config=config, stream_mode="values"):
#         event["messages"][-1].pretty_print()

#     print("---"*33)



class OutputFormat(BaseModel):
    preference: str
    sentence_preference_revealed: str

class TelegramPreferences(BaseModel):
    preferred_encoding: Optional[List[OutputFormat]] = None
    favorite_telegram_operators: Optional[List[OutputFormat]] = None
    preferred_telegram_paper: Optional[List[OutputFormat]] = None

class MorseCode(BaseModel):
    preferred_key_type: Optional[List[OutputFormat]] = None
    favorite_morse_abbreviations: Optional[List[OutputFormat]] = None

class Semaphore(BaseModel):
    preferred_flag_color: Optional[List[OutputFormat]] = None
    semaphore_skill_level: Optional[List[OutputFormat]] = None

class TrustFallPreferences(BaseModel):
    preferred_fall_height: Optional[List[OutputFormat]] = None
    trust_level: Optional[List[OutputFormat]] = None
    preferred_catching_technique: Optional[List[OutputFormat]] = None

class CommunicationPreferences(BaseModel):
    telegram: TelegramPreferences
    morse_code: MorseCode
    semaphore: Semaphore

class UserPreferences(BaseModel):
    communication_preferences: CommunicationPreferences
    trust_fall_preferences: TrustFallPreferences

class TelegramAndTrustFallPreferences(BaseModel):
    pertinent_user_preferences: UserPreferences


model_name = input("Enter model name: ")
llm = ChatOpenAI(model=model_name)
llm_with_structure = llm.with_structured_output(TelegramAndTrustFallPreferences)

# Conversation
conversation = """Operator: How may I assist with your telegram, sir?
Customer: I need to send a message about our trust fall exercise.
Operator: Certainly. Morse code or standard encoding?
Customer: Morse, please. I love using a straight key.
Operator: Excellent. What's your message?
Customer: Tell him I'm ready for a higher fall, and I prefer the diamond formation for catching.
Operator: Done. Shall I use our "Daredevil" paper for this daring message?
Customer: Perfect! Send it by your fastest carrier pigeon.
Operator: It'll be there within the hour, sir."""

# # Invoke the model
# try:
#     # gpt-4o-mini could parse this without trustcall. 3.5-turbo could not but I stopped it after 30 secs.
#     response = llm_with_structure.invoke(f"""Extract the preferences from the following conversation:
#     <convo>
#     {conversation}
#     </convo>""")
#     print(f"No validation error")
#     print(response)
# except ValidationError as e:
#     print(e)


extractor = create_extractor(
    llm,
    tools=[TelegramAndTrustFallPreferences],
    tool_choice="TelegramAndTrustFallPreferences"
)

# Also errored with 3.5-turbo and gpt-4o-mini.
# gpt-4o worked.
result = extractor.invoke(f"""Extract the preferences from the following conversation:
    <convo>
    {conversation}
    </convo>""")

for m in result["messages"]:
    m.pretty_print()

schema = result["responses"][0]
print(schema)

schema.model_dump()

print(result["response_metadata"])
