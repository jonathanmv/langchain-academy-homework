from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)


from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

llm_with_tools = llm.bind_tools(tools)

system_message = SystemMessage(content="You are a helpful assistant that can multiply, sum, and divide numbers.")

def assistant_node(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}


builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
# builder.add_edge("assistant", END) # This edge is not needed because the tools_condition will go to tools or END.

memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])
# By setting interrupt_before, the graph execution is interrupted before the tools node is executed.
# We can use graph.stream to execute the graph until the interruption point.

user_input = input("Specify an arithmetic operation: ")
config = {"configurable": {"thread_id": 1 }}
for chunk in graph.stream({"messages":[HumanMessage(content=user_input)]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# You could also use graph.invoke and it will stop at the interruption point.
# response = graph.invoke({"messages":["What is 2 times 3?"]}, config)
# for m in response["messages"]:
#     m.pretty_print()

# You can get the current state of the graph using graph.get_state
state = graph.get_state(config)
# Get the keys of the state
print("State keys:")
print(state.values.keys())
# You can check for the next node by checking state.next
print("Next node:")
print(state.next)

user_input = input("\nExecute operation? (y/n)")
if user_input == "y":
    # Passing None will continue the graph execution from the interruption point.
    for chunk in graph.stream(None, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
else:
    print("Bye!")
