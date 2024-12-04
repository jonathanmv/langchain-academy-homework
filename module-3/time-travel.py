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
graph = builder.compile(checkpointer=memory)

user_input = input("Specify an arithmetic operation: ")
config = {"configurable": {"thread_id": 1 }}
for chunk in graph.stream({"messages":[HumanMessage(content=user_input)]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# The graph runs until the end because we did not set interrupt_before.
# You can get the current state of the graph using graph.get_state
state = graph.get_state(config)

# And you can also get the states until this point using graph.get_state_history
all_states = [s for s in graph.get_state_history(config)]


def section(title):
    print("\n")
    print("="*80)
    print(title)
    print("="*80)
    print("\n")

section("State history")
# The states are in reverse order (most recent first)
for i, s in enumerate(all_states):
    print("="*80)
    last_message = s.values["messages"][-1].content if s.values["messages"] else "<empty>"
    print(f"State {i}: {last_message}")
    print(f"Metadata step: {s.metadata['step']}")
    print(f"Config: {s.config}")
    print(f"Next node: {s.next}")

section("State at step 0 (human message)")
# You can get the config of a given state snapshot by accessing s.config
human_message_state = all_states[-2]
print(f"State of metadata.step 0 (human message)")
print(f"Message: {human_message_state.values['messages'][-1].content}")
print(f"Config: {human_message_state.config}")

section("Replaying the graph from the human message state")
# You can replay the graph from a given state by passing the config of that state to the graph.stream method.
# No new llm calls are made because the graph already ran and the state is saved in the memory.
for chunk in graph.stream(None, human_message_state.config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


section("Forking the graph")

# You can fork the graph by updating the state of a given config.
# Let's fork the message to include a new operation.

forked_config = graph.update_state(
    human_message_state.config,
    {"messages": [
        *human_message_state.values["messages"],
        HumanMessage(content="And all of that times 2")
    ]}
)

print(f"Forked config: {forked_config}")
all_states = [s for s in graph.get_state_history(forked_config)]

section("Forked state history")
for i, s in enumerate(all_states):
    print(f"State {i}: {s.values['messages'][-1].content}")

section("Replaying the forked graph")
for chunk in graph.stream(None, forked_config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
