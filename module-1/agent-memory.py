from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


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


llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)



system_message = SystemMessage(content="You are a helpful assistant that can multiply, sum, and divide numbers.")

def agent_node(state: MessagesState) -> MessagesState:
    messages = [system_message] + state["messages"]
    return { "messages": [llm_with_tools.invoke(messages)] }


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

print(react_graph_memory.get_graph().draw_ascii())
#         +-----------+
#         | __start__ |
#         +-----------+
#               *
#               *
#               *
#           +-------+
#           | agent |
#           +-------+
#          *         .
#        **           ..
#       *               .
# +-------+         +---------+
# | tools |         | __end__ |
# +-------+         +---------+


config = { "configurable": { "thread_id": "1" }}
messages = [HumanMessage(content="What is 3 times 4?")]

result = react_graph_memory.invoke({ "messages": messages}, config)

for msg in result["messages"]:
    msg.pretty_print()

# Add more messages to test memory
messages = [HumanMessage(content="how much is that divided by 6?")] # 2
result = react_graph_memory.invoke({ "messages": messages}, config)

for msg in result["messages"]:
    msg.pretty_print()
