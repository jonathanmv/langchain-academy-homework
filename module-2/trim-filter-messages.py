from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, trim_messages
from langgraph.graph import StateGraph, END, START, MessagesState


llm = ChatOpenAI(model="gpt-4o")

def chat_node(state: MessagesState):
    return { "messages": llm.invoke(state["messages"]) }

builder = StateGraph(MessagesState)
builder.add_node("chat_node", chat_node)
builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)

graph = builder.compile()

print(graph.get_graph().draw_ascii())


# print("invoking with messages = ['what date is today?']")
# result = graph.invoke({ "messages": ["what date is today?"] })

# for message in result["messages"]:
#     message.pretty_print()


def filter_messages_node(state: MessagesState):
    # Delete all messages except the two most recent ones
    messages_to_delete = [RemoveMessage(id=message.id) for message in state["messages"][:-2]]
    return { "messages": messages_to_delete}

builder = StateGraph(MessagesState)
builder.add_node("filter_messages_node", filter_messages_node)
builder.add_node("chat_node", chat_node)

builder.add_edge(START, "filter_messages_node")
builder.add_edge("filter_messages_node", "chat_node")
builder.add_edge("chat_node", END)

graph = builder.compile()

print("\n\nGraph with filter messages node:")
print(graph.get_graph().draw_ascii())
#       +-----------+
#       | __start__ |
#       +-----------+
#             *
#             *
#             *
# +----------------------+
# | filter_messages_node |
# +----------------------+
#             *
#             *
#             *
#       +-----------+
#       | chat_node |
#       +-----------+
#             *
#             *
#             *
#       +---------+
#       | __end__ |
#       +---------+

messages = [
    AIMessage(content="hello", id="1"),
    HumanMessage(content="hi, you will forget this message", id="2"),
    AIMessage(content="I'm sorry, but I can't provide real-time information. Please check your device or calendar for today's date.", id="3"),
    HumanMessage(content=f"I see, ok, today is {datetime.now().strftime('%Y-%m-%d')}. So what date is tomorrow?", id="4")
]

# result = graph.invoke({ "messages": messages })

# print("\n\nResult calling graph with filter messages node:")
# for message in result["messages"]:
#     message.pretty_print()


def chat_with_trimmed_messages_node(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        max_tokens=30,
        strategy="last",
        token_counter=llm,
        allow_partial=False
    )
    return { "messages": [llm.invoke(messages)] }

builder = StateGraph(MessagesState)
builder.add_node("chat_with_trimmed_messages_node", chat_with_trimmed_messages_node)

builder.add_edge(START, "chat_with_trimmed_messages_node")
builder.add_edge("chat_with_trimmed_messages_node", END)

graph = builder.compile()

print("\n\nGraph with chat with trimmed messages node:")
print(graph.get_graph().draw_ascii())
#            +-----------+
#            | __start__ |
#            +-----------+
#                   *
#                   *
#                   *
# +---------------------------------+
# | chat_with_trimmed_messages_node |
# +---------------------------------+
#                   *
#                   *
#                   *
#             +---------+
#             | __end__ |
#             +---------+

print("\n\nTrimmed messages:")
trimmed_messages = trim_messages(messages, max_tokens=60, strategy="last", token_counter=llm, allow_partial=False)
for message in trimmed_messages:
    message.pretty_print()

# Trimmed messages:
# ================================== Ai Message ==================================

# I'm sorry, but I can't provide real-time information. Please check your device or calendar for today's date.
# ================================ Human Message =================================

# I see, ok, today is 2024-11-26. So what date is tomorrow?

# print("\n\nResult calling graph with chat with trimmed messages node:")
# result = graph.invoke({ "messages": messages })

# for message in result["messages"]:
#     message.pretty_print()