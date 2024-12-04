from typing import TypedDict

from langgraph.errors import NodeInterrupt
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class DynamicBreakpointState(TypedDict):
    input: str

def step_1(state):
    print("Step 1")
    return state

def step_2(state):
    print("Step 2")
    length = len(state["input"])
    if length > 5:
        print(f"Interrupting because input is {length} characters long...")
        raise NodeInterrupt(f"Input must be less than 6 characters")
    return state

def step_3(state):
    print("Step 3")
    return state

builder = StateGraph(DynamicBreakpointState)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable":{"thread_id": "1"}}

def next_node():
    return (graph.get_state(config).next or (None,))[0]

while True:
    user_input = input("\nEnter a string or q to quit: ")
    if user_input == "q":
        print("Bye!")
        break
    for chunk in graph.stream({"input": user_input}, config, stream_mode="values"):
        print(f"Executing node '{next_node()}' with input '{chunk['input']}'")

    # Interrupted or finished
    next_node = next_node()
    print(f"\nNext node: {next_node}")
    if next_node is None:
        print("Your string was valid. Bye!")
        break
