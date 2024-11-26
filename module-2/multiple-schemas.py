from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END

class OverallState(BaseModel):
    foo: int

class PrivateState(BaseModel):
    bar: int

def node_1(state: OverallState) -> PrivateState:
    return { "bar": state.foo + 1}

def node_2(state: PrivateState) -> OverallState:
    return { "foo": state.bar + 1 }

builder = StateGraph(OverallState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

graph = builder.compile()

print("\nPrivate state graph:")
print(graph.get_graph().draw_ascii())
# +-----------+
# | __start__ |
# +-----------+
#       *
#       *
#       *
#   +--------+
#   | node_1 |
#   +--------+
#       *
#       *
#       *
#   +--------+
#   | node_2 |
#   +--------+
#       *
#       *
#       *
#  +---------+
#  | __end__ |
#  +---------+

print("invoking with foo = 0")
result = graph.invoke({ "foo": 0 })

print(f"result: {result}\n\n")
# { "foo": 2 }

### Input/output schemas

class InputSchema(BaseModel):
    question: str

class OutputSchema(BaseModel):
    answer: str

class OverallStateTwo(BaseModel):
    question: str
    answer: str
    notes: str

def first_node(state: InputSchema):
    return { "answer": "bye", "notes": "... his name is jonathanmv" }

def second_node(state: OverallStateTwo) -> OutputSchema:
    return { "answer": "bye jonathanmv" }

builder = StateGraph(OverallStateTwo, input=InputSchema, output=OutputSchema)
builder.add_node("first_node", first_node)
builder.add_node("second_node", second_node)

builder.add_edge(START, "first_node")
builder.add_edge("first_node", "second_node")
builder.add_edge("second_node", END)

graph = builder.compile()

print("\nInput/output graph:")
print(graph.get_graph().draw_ascii())
#  +-----------+
#  | __start__ |
#  +-----------+
#         *
#         *
#         *
# +------------+
# | first_node |
# +------------+
#         *
#         *
#         *
# +-------------+
# | second_node |
# +-------------+
#         *
#         *
#         *
#   +---------+
#   | __end__ |
#   +---------+

print("invoking with question = 'hello'")
result = graph.invoke({ "question": "hello" })

print(f"result: {result}\n\n")
# { "answer": "bye jonathanmv" }