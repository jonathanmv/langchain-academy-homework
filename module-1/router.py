from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

def multiply(a: int, b: int) -> int:
    """Multiply a and b

    Args:
        a: int
        b: int
    """
    return a * b

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([multiply])

def tool_calling_llm_node(state: MessagesState) -> MessagesState:
    """Extracts the function and arguments from the messages"""
    return { "messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm_node)
# "tools" needs to be the name of the node
# otherwise tools_condition will not work
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
# tools_condition checks if the last message is the response of a tool call
# if it's a tool call, it will route to the "tools" node (hardcoded)
# if it's not a tool call, it will route to the "END" node (hardcoded)
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)

graph = builder.compile()

print(graph.get_graph().draw_ascii())

#         +-----------+
#         | __start__ |
#         +-----------+
#               *
#               *
#               *
#     +------------------+
#     | tool_calling_llm |
#     +------------------+
#          ..        ..
#        ..            .
#       .               ..
# +-------+               .
# | tools |             ..
# +-------+            .
#          **        ..
#            **    ..
#              *  .
#          +---------+
#          | __end__ |
#          +---------+

print("\nTesting non-multiplication message")
result = graph.invoke({ "messages": [HumanMessage(content="Hi, this has nothing to do with multiplying.")] })
for msg in result["messages"]:
    msg.pretty_print()

print("\nTesting multiplication message")
result = graph.invoke({ "messages": [HumanMessage(content="What is 2 times 3?")] })
for msg in result["messages"]:
    msg.pretty_print()


