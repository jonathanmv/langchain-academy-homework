import uuid
import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, MessagesState
from langgraph.graph import START, END
from langgraph.checkpoint.memory import MemorySaver

class InterpretationState(MessagesState):
    interpretation: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def interpretation_node(state: InterpretationState):
    messages = state.get('messages', [])
    if len(messages) < 4:
        return state

    print("Interpreting...")

    interpretation = state.get('interpretation', 'No interpretation yet')

    prompt = [
        SystemMessage(content=f'Interpretation: {interpretation}'),
        SystemMessage(content="Please update the interpretation for the conversation"),
    ]
    interpretation = llm.invoke(messages + prompt)
    # Keep the two most recent messages
    return { 'messages': messages[-2:], 'interpretation': interpretation.content }

def chat_node(state: InterpretationState):
    messages = state.get('messages', [])
    interpretation = state.get('interpretation', 'No interpretation yet')

    prompt = [
        SystemMessage(content=f'Interpretation: {interpretation}'),
        SystemMessage(content="Please continue the conversation taking into account the interpretation"),
    ]
    response = llm.invoke(prompt + messages)

    return { 'messages': response, 'interpretation': interpretation }

def interpret_based_on_messages(state: InterpretationState):
    messages = state.get('messages', [])
    if len(messages) > 3:
        return 'interpretation_node'

    return END


builder = StateGraph(InterpretationState)
builder.add_node('chat_node', chat_node)
builder.add_node('interpretation_node', interpretation_node)

builder.add_edge(START, 'chat_node')
builder.add_conditional_edges('chat_node', interpret_based_on_messages)
builder.add_edge('interpretation_node', END)

async def run_graph(graph, user_input, config, streaming_mode):
    input = { 'messages': [HumanMessage(content=user_input)] }
    match streaming_mode:
        case 'updates':
            # gets state after the first node is ran
            print(f"Streaming updates...")
            for chunk in graph.stream(input, config, stream_mode=streaming_mode):
                chunk['conversation']['messages'].pretty_print()
        case 'values':
            # gets the full state after the first node is ran
            print(f"Streaming values...")
            for event in graph.stream(input, config, stream_mode=streaming_mode):
                for m in event["messages"]:
                    m.pretty_print()
        case 'events':
            print(f"Streaming events...")
            async for event in graph.astream_events(input, config, version="v2"):
                if event['event'] != 'on_chat_model_stream':
                    continue

                if event['metadata'].get('langgraph_node', '') != 'chat_node':
                    continue

                data = event['data']
                print(data["chunk"].content, end="•")
        case 'off':
            result = await graph.ainvoke(input, config)
            for m in result.get('messages', []):
                m.pretty_print()

async def main():
    thread_id = str(uuid.uuid4())
    streaming_mode: 'updates' | 'values' | 'events' | 'off' = 'off'

    db_memory = MemorySaver()
    graph = builder.compile(checkpointer=db_memory)

    while True:
        config = { 'configurable': { 'thread_id': thread_id } }
        user_input = input(f"You ({thread_id}) (streaming: {streaming_mode}): ")
        if user_input == 'exit':
            print("Exiting...")
            break

        if user_input == 'clear':
            thread_id = str(uuid.uuid4())
            print(f"Thread changed to {thread_id}")
            continue

        if user_input == 'interpretation':
            state = await graph.aget_state(config)
            print(state.values.get('interpretation', 'No interpretation yet'))
            continue

        if user_input.startswith('stream'):
            if user_input.split(' ')[1] in ['updates', 'values', 'events', 'off']:
                print(f"Streaming mode changed to {user_input.split(' ')[1]}")
                streaming_mode = user_input.split(' ')[1]
            else:
                print("Invalid streaming mode. Please use 'stream_updates', 'stream_values', 'stream_events', or 'stream_off'.")
            continue

        if user_input.startswith('thread'):
            thread_id = user_input.split(' ')[1]
            print(f"Thread changed to {thread_id}")
            continue

        print(f"Running graph...")
        await run_graph(graph, user_input, config, streaming_mode)

if __name__ == '__main__':
    asyncio.run(main())


# After a few tries, I couldn't run streaming with SQLite. It asked me use AsyncSqliteSaver.
# But I got different errors. I'm now using memory for the checkpointer.