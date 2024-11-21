from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from pprint import pprint
import random
from typing_extensions import TypedDict
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import MessagesState as LangGraphMessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages

# Message: text + role (human, ai, system, tool)
messages = [
    SystemMessage(content="jonathanmv is learning about langgraph"),
    AIMessage(content="Hi jonathanmv, how can I help you today?"),
    HumanMessage(content="I'm learning about langgraph. Can you tell me more about it?")
]

for msg in messages:
    msg.pretty_print()

llm = ChatOpenAI(model="gpt-4o")

# response = llm.invoke(messages) # returns a single AIMessage
# print(type(response))
# print(response.response_metadata) # contains a "finish_reason" (stop, tool_calls)
# print(response.content)

def multiply(a: int, b: int) -> int:
    """
    Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])

# tool_call = llm_with_tools.invoke([HumanMessage(content="What is 2 * 3?")])
# print("tool_call")
# print(type(tool_call))
# print(tool_call)
# print(tool_call.response_metadata["tool_calls"])

class OverridesMessagesState(TypedDict):
    messages: list[AnyMessage]

class AppendsMessagesState(TypedDict):
    # add_messages is a reducer that merges messages
    # so we can append to the list of messages
    # new messages get an id. You can use that id to override a message
    messages: Annotated[list[AnyMessage], add_messages]

class InheritedMessagesState(LangGraphMessagesState):
    # LangGraphMessagesState contains a "messages" field
    # We can add other fields to the state here
    pass

StateClass = Annotated[OverridesMessagesState, AppendsMessagesState, InheritedMessagesState]

overrides_messages_state = OverridesMessagesState(messages=messages)
print("overrides_messages_state")
print(overrides_messages_state)

appends_messages_state = AppendsMessagesState(messages=messages)
print("appends_messages_state")
print(appends_messages_state)

inherited_messages_state = InheritedMessagesState(messages=messages)
print("inherited_messages_state")
print(inherited_messages_state)

# so far, they all print the same: a dict with a "messages" field

def random_int_node(state: StateClass) -> StateClass:
    """
    Generate a random integer.
    """
    random_int = random.randint(0, 100)
    return { "messages": [SystemMessage(content=f"The random integer is {random_int}.")] }

graphs = {
    "OverridesMessagesState": None,
    "AppendsMessagesState": None,
    "InheritedMessagesState": None
}

for StateClass in [OverridesMessagesState, AppendsMessagesState, InheritedMessagesState]:
    builder = StateGraph(StateClass)
    builder.add_node("random_int", random_int_node)
    builder.add_edge(START, "random_int")
    builder.add_edge("random_int", END)
    graph = builder.compile()
    graphs[StateClass.__name__] = graph
    print(f"\nInvoking graph with State: {StateClass.__name__}")
    initial_state = { "messages": [HumanMessage(content="Generate a random integer.")] }
    print(f"Initial state: {initial_state}")
    final_state = graph.invoke(initial_state)
    print(f"Final state: {final_state}")

# Let's use the AppendsMessagesState to create a graph that generates two random integers
# and then multiplies them

class MultiplyState(LangGraphMessagesState):
    a: int
    b: int
    result: int

def extract_multiply_args(state: MultiplyState) -> MultiplyState:
    result = llm_with_tools.invoke(state["messages"])
    args = result.tool_calls[0]["args"]
    return { "messages": [llm_with_tools.invoke(state["messages"])], "a": args["a"], "b": args["b"] }

def multiply_node(state: MultiplyState) -> MultiplyState:
    result = multiply(state["a"], state["b"])
    return { "messages": [SystemMessage(content=f"The result of multiplying {state['a']} and {state['b']} is {result}.")], "result": result }

builder = StateGraph(AppendsMessagesState)
builder.add_node("random_int_1", random_int_node)
builder.add_node("random_int_2", random_int_node)
builder.add_node("extract_multiply_args", extract_multiply_args)
builder.add_node("multiply", multiply_node)
builder.add_edge(START, "random_int_1")
builder.add_edge(START, "random_int_2")
builder.add_edge("random_int_1", "extract_multiply_args")
builder.add_edge("random_int_2", "extract_multiply_args")
builder.add_edge("extract_multiply_args", "multiply")
builder.add_edge("multiply", END)

graph = builder.compile()

print(graph.get_graph().draw_ascii())
result = graph.invoke({ "messages": [HumanMessage(content="Generate two random integers and multiply them.")] })
for msg in result["messages"]:
    msg.pretty_print()
