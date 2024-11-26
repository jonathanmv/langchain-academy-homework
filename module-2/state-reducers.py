from operator import add
from typing import Annotated
from pydantic import BaseModel

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.errors import InvalidUpdateError

from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage

class State(BaseModel):
    foo: int

def increment_foo_node(state: State) -> State:
    return { "foo": state.foo + 1 }


builder = StateGraph(State)
builder.add_node("increment_foo", increment_foo_node)
builder.add_edge(START, "increment_foo")
builder.add_edge("increment_foo", END)

graph = builder.compile()

print(graph.get_graph().draw_ascii())
#   +-----------+
#   | __start__ |
#   +-----------+
#         *
#         *
#         *
# +---------------+
# | increment_foo |
# +---------------+
#         *
#         *
#         *
#    +---------+
#    | __end__ |
#    +---------+

result = graph.invoke({ "foo": 0 })

print(result)
# { "foo": 1 }


# You can remove nodes or edges, so let's create the builder again
builder = StateGraph(State)

builder.add_node("increment_foo", increment_foo_node)
# add nodes that execute in parallel and update the same key in the state
builder.add_node("also_increment_foo",increment_foo_node)
builder.add_node("increment_foo_once_more",increment_foo_node)

builder.add_edge(START, "increment_foo")
builder.add_edge("increment_foo", "also_increment_foo")
builder.add_edge("increment_foo", "increment_foo_once_more")
builder.add_edge("also_increment_foo", END)
builder.add_edge("increment_foo_once_more", END)

graph = builder.compile()

print(graph.get_graph().draw_ascii())
#                       +-----------+
#                       | __start__ |
#                       +-----------+
#                              *
#                              *
#                              *
#                     +---------------+
#                     | increment_foo |
#                     +---------------+
#                    ***               ***
#                 ***                     ***
#               **                           **
# +--------------------+           +-------------------------+
# | also_increment_foo |           | increment_foo_once_more |
# +--------------------+           +-------------------------+
#                    ***               ***
#                       ***         ***
#                          **     **
#                        +---------+
#                        | __end__ |
#                        +---------+

try:
    result = graph.invoke({ "foo": 0 })
except InvalidUpdateError as e:
    # langgraph.errors.InvalidUpdateError: At key 'foo': Can receive only one value per step.
    # Use an Annotated key to handle multiple values.
    print("Expected InvalidUpdateError:", e)

class AnnotatedState(BaseModel):
    foo: Annotated[list[int], add]

# If I define the type of the state, I won't be able to use the increment_foo_reducer in a graph with a different state
# even if the other state also has a foo key. It seems like langgraph keeps track of the type of each key in the state.
# I was using `Annotated[list[int], add]` in one state and `Annotated[list[int], merge_lists_reducer]` in another state.
# I would get a ValueError: Channel 'foo' already with a different type
def increment_foo_reducer_node(state):
    return { "foo": [state.foo[-1] + 1] }

builder = StateGraph(AnnotatedState)
builder.add_node("increment_foo_reducer", increment_foo_reducer_node)
builder.add_node("increment_foo_once_more", increment_foo_reducer_node)
builder.add_node("also_increment_foo", increment_foo_reducer_node)

builder.add_edge(START, "increment_foo_reducer")
builder.add_edge("increment_foo_reducer", "also_increment_foo")
builder.add_edge("increment_foo_reducer", "increment_foo_once_more")
builder.add_edge("also_increment_foo", END)
builder.add_edge("increment_foo_once_more", END)

graph = builder.compile()

result = graph.invoke({ "foo": [0] })

print("AnnotatedState result:")
print(result)
# { "foo": [0, 1, 2, 2] }

try:
    # the add operator expects a list
    result = graph.invoke({ "foo": 0 })
except TypeError as e:
    print("Expected TypeError:", e)

def merge_lists_reducer(left: list | int | None, right: list | int | None) -> list[int]:
    if isinstance(left, int):
        left = [left]
    if isinstance(right, int):
        right = [right]

    left = left or []
    right = right or []

    return left + right

class CustomReducerState(BaseModel):
    foo: Annotated[list[int], merge_lists_reducer]

builder = StateGraph(CustomReducerState)
builder.add_node("increment_foo_reducer", increment_foo_reducer_node)
builder.add_node("increment_foo_once_more", increment_foo_reducer_node)
builder.add_node("also_increment_foo", increment_foo_reducer_node)

builder.add_edge(START, "increment_foo_reducer")
builder.add_edge("increment_foo_reducer", "also_increment_foo")
builder.add_edge("increment_foo_reducer", "increment_foo_once_more")
builder.add_edge("also_increment_foo", END)
builder.add_edge("increment_foo_once_more", END)

graph = builder.compile()

result = graph.invoke({ "foo": 0 }) # invoking with int should still work

print("CustomReducerState result:")
print(result)
# { "foo": [0, 1, 2, 2] }


# MessagesState uses `add_message` in the "messages" key
class ExtendedMessagesState(MessagesState):
    # messages: Annotated[list[langchain_core.messages.AnyMessage], add_messages]
    # add more keys here if needed
    pass

# the add_messages reducer allows us to add, edit, or remove messages in the state

print("ADDING A MESSAGE")
initial_messages = [
    HumanMessage(content="Hello, world!"),
    AIMessage(content="Hello, human!")
]

new_message = HumanMessage(content="How are you?")

for message in add_messages(initial_messages, new_message):
    print(f"Message Id: {message.id}") # if undefined, add_messages will assign an id
    message.pretty_print()

print("\nEDITING A MESSAGE")
initial_messages = [
    HumanMessage(content="Hello, world!", id="1"),
    AIMessage(content="Hello, human!", id="2")
]

edited_message = HumanMessage(content="This is not an AI message anymore!", id="2")

for message in add_messages(initial_messages, edited_message):
    print(f"Message Id: {message.id}")
    message.pretty_print()

print("\nREMOVING A MESSAGE")
initial_messages = [
    HumanMessage(content="Hello, world!", id="1"),
    AIMessage(content="Hello, human!", id="2")
]

removed_message = RemoveMessage(id="1")

for message in add_messages(initial_messages, removed_message):
    print(f"Message Id: {message.id}")
    message.pretty_print()
