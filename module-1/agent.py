from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


def multiply(a: int, b: int) -> int:
    """Multiply a and b

    Args:
        a: int
        b: int
    """
    return a * b

def sum(a: int, b: int) -> int:
    """Sum a and b

    Args:
        a: int
        b: int
    """
    return a + b

def divide(a: int, b: int) -> int:
    """Divide a by b

    Args:
        a: int
        b: int
    """
    return a / b

tools = [multiply, sum, divide]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: MessagesState) -> MessagesState:
    messages = [
        SystemMessage(content="You are a helpful assistant that can multiply, sum, and divide numbers."),
        *state["messages"]
    ]
    return { "messages": [llm_with_tools.invoke(messages)]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
# END is not needed because tools_condition will route to END if no tool call is found

graph = builder.compile()

print(graph.get_graph().draw_ascii())

print("\nTesting non-multiplication message")
result = graph.invoke({ "messages": [HumanMessage(content="Hi, this has nothing to do with math!")] })
for msg in result["messages"]:
    msg.pretty_print()

print("\nTesting arithmetic message")

result = graph.invoke({ "messages": [
    HumanMessage(content="multiply 2 and 3 and "), # 6
    HumanMessage(content="substract 1 to the output."), # 5
    HumanMessage(content="Now add 10 to that result and "), # 15
    HumanMessage(content="finally divide by 3."), # 5
] })
for msg in result["messages"]:
    msg.pretty_print()
