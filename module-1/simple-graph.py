import random
from typing import Literal
from langgraph.graph import StateGraph, START, END

# State: Defines the input schema for all nodes in the graph
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str

# Nodes: Just python functions that take the state as input and return a new state
def node_1(state: State) -> State:
    print(f"{state['graph_state']} >> node 1 >> {state['graph_state']} is ")
    return { "graph_state": state["graph_state"] + " is " }

def node_2(state: State) -> State:
    print(f"{state['graph_state']} >> node 2 >> {state['graph_state']}awesome")
    return { "graph_state": state["graph_state"] + "awesome" }

def node_3(state: State) -> State:
    print(f"{state['graph_state']} >> node 3 >> {state['graph_state']}boring")
    return { "graph_state": state["graph_state"] + "boring" }

# Edges: Connects nodes together. Conditional edges are functions that take the state and return the next node to execute.
def flip_coin(state: State) -> Literal["awesome", "boring"]:
    print(f"{state['graph_state']} >> flip_coin >> (node 2 or node 3)")
    # You can use the state to make decisions
    return random.choice(["awesome", "boring"])

# Graph: A collection of nodes and edges. START and END are special nodes to initiate and end the graph.
builder = StateGraph(State)
# Add nodes: maps a name to a node function
builder.add_node("flip_coin", node_1)
builder.add_node("awesome", node_2)
builder.add_node("boring", node_3)

# Add edges
# add_edge(from_node, to_node): maps a node name to the next node name
builder.add_edge(START, "flip_coin")
# add_conditional_edges(from_node, conditional_function): maps a node name to a conditional function that returns the next node name
builder.add_conditional_edges("flip_coin", flip_coin)
builder.add_edge("awesome", END)
builder.add_edge("boring", END)

# Compile the graph to make validations
graph = builder.compile()

# Draw the graph
print("Running the following graph:")
print(graph.get_graph().draw_ascii())

#          +-----------+
#          | __start__ |
#          +-----------+
#                 *
#                 *
#                 *
#          +-----------+
#          | flip_coin |
#          +-----------+
#           ..        ..
#         ..            ..
#        .                .
# +---------+          +--------+
# | awesome |          | boring |
# +---------+          +--------+
#           **        **
#             **    **
#               *  *
#           +---------+
#           | __end__ |
#           +---------+

# Request user input
user_input = input("Enter an initial state: ")

# Invoke the graph
print("Invoking the graph with the following state:")
print({"graph_state": user_input})
print("\nResult:")
print(graph.invoke({"graph_state": user_input}))
