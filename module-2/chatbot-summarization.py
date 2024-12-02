import uuid
from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage

from langgraph.graph import StateGraph, MessagesState
from langgraph.graph import START, END
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def conversation_node(state: MessagesState):
    summary = state.get("summary", "")
    messages = state.get("messages", [])

    if summary:
        summary_message = SystemMessage(content=f"Consider this summary of the conversation when responding: {summary}")
        messages = [summary_message] + messages

    return { "messages": [llm.invoke(messages)] }

def summarize_if_long_conversation_node(state: MessagesState):
    messages = state.get("messages", [])
    if len(messages) > 5:
        print("\nSummarizing conversation...\n")
        return "summarize_conversation_node"

    # End this cycle. Use external memory to store messages.
    # New cycles will call the conversation_node with the new messages.
    # If we return "conversation_node", it will stay in an infinite loop.
    return END

def summarize_conversation_node(state: MessagesState):
    summary = state.get("summary", "")
    messages = state.get("messages", [])
    if summary:
        summary_message = SystemMessage(content=f"Summarize the conversation above taking into account the current summary: {summary}")
        summary_result = llm.invoke(messages + [summary_message])
        summary = summary_result.content
    else:
        summary_message = SystemMessage(content="Summarize the conversation above.")
        summary_result = llm.invoke(messages + [summary_message])
        summary = summary_result.content

    print(f"\nSummary: {summary}\n")

    # Now that we have a summary, we can remove most of the messages
    messages = [RemoveMessage(id=m.id) for m in messages[:-2]]

    return { "messages": messages, "summary": summary }

builder = StateGraph(MessagesState)

builder.add_node("conversation_node", conversation_node)
builder.add_node("summarize_conversation_node", summarize_conversation_node)

builder.add_edge(START, "conversation_node")
builder.add_conditional_edges("conversation_node", summarize_if_long_conversation_node)
builder.add_edge("summarize_conversation_node", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Get a graph's state with its thread id
# config = { "configurable": { "thread_id": "1" }}
# graph.get_state(config).values.get("summary", "")

def chat():
    thread_id = str(uuid.uuid4())

    print("Chat started. Commands: 'exit', 'summary', 'clear'")

    while True:
        config = { "configurable": { "thread_id": thread_id }}
        user_input = input("\nYou: ")

        if user_input.lower() == 'exit':
            break

        if user_input.lower() == 'summary':
            print(f"\nSummary: {graph.get_state(config).values.get('summary', '')}\n")
            continue

        if user_input.lower() == 'clear':
            print(f"\nStarting new chat...\n")
            thread_id = str(uuid.uuid4())
            continue

        print("\nThinking...\n")
        result = graph.invoke({ "messages": [HumanMessage(content=user_input)] }, config)

        for message in result["messages"][-1:]:
            print(f"Assistant: {message.content}\n")

if __name__ == "__main__":
    chat()
