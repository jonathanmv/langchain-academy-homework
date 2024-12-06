from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)


import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader

class State(TypedDict):
    # This operator appends. If two parallel nodes update the same key, we get an error.
    state: Annotated[list, operator.add]

def append(string: str):
    def state_update(state):
        print(f"Appending {string} to {state}")
        return {"state": [string]}
    return state_update

builder = StateGraph(State)

builder.add_node('a', append("a"))
builder.add_node('b', append("b"))
builder.add_node('b2', append("b2"))
builder.add_node('c', append("c"))
builder.add_node('d', append("d"))

builder.add_edge(START, 'a')
builder.add_edge('a', 'c')
builder.add_edge('a', 'b')
builder.add_edge('b', 'b2')
builder.add_edge(['b2', 'c'], 'd') # Interesting how to fan-in the branches. The opposite didn't work.
builder.add_edge('d', END)

graph = builder.compile()


print(graph.get_graph().draw_ascii())

print(graph.invoke({"state": ['->']}))
# {'state': ['->', 'a', 'b', 'c', 'b2', 'd']}
# Notice that 'b2' happens after 'c'. It seems like the state operator is adding nodes in alphabetical order
# even though the nodes run in different orders.
# In the example below, we can see that e runs before c, but when b2 is called, c is before e.
# Appending a to {'state': ['->']}
# Appending e to {'state': ['->', 'a']}
# Appending c to {'state': ['->', 'a']}
# Appending b2 to {'state': ['->', 'a', 'c', 'e']}
# Appending d to {'state': ['->', 'a', 'c', 'e', 'b2']}

# We can change that by using our own operator (reducer).

def sorting_reducer(left, right):
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return sorted(left + right, reverse=False)

class SortedState(TypedDict):
    state: Annotated[list, sorting_reducer]



builder = StateGraph(SortedState)

builder.add_node('a', append("a"))
builder.add_node('b', append("b"))
builder.add_node('b2', append("b2"))
builder.add_node('c', append("c"))
builder.add_node('d', append("d"))

builder.add_edge(START, 'a')
builder.add_edge('a', 'c')
builder.add_edge('a', 'b')
builder.add_edge('b', 'b2')
builder.add_edge(['b2', 'c'], 'd') # Interesting how to fan-in the branches. The opposite didn't work.
builder.add_edge('d', END)

graph = builder.compile()

print("\nSorted graph:")
print(graph.invoke({"state": ['->']}))
# {'state': ['->', 'a', 'b', 'b2', 'c', 'd']}
# Even though the state looks like we want, c still runs before b2.
# Appending a to {'state': ['->']}
# Appending b to {'state': ['->', 'a']}
# Appending c to {'state': ['->', 'a']}
# Appending b2 to {'state': ['->', 'a', 'b', 'c']}
# Appending d to {'state': ['->', 'a', 'b', 'b2', 'c']}
# {'state': ['->', 'a', 'b', 'b2', 'c', 'd']}

class QuestionState(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def tavily_search_node(state: QuestionState):
    search = TavilySearchResults(max_results=3)
    results = search.invoke(state['question'])
    formatted_results = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in results
        ]
    )

    return {"context": [formatted_results]}


def wikipedia_search_node(state: QuestionState):
    results = WikipediaLoader(query=state["question"], load_max_docs=2).load()
    formatted_results = "\n\n---\n\n".join(
        [
            f'<Document href="{doc.metadata["source"]}" page="{doc.metadata.get("page","")}"/>\n{doc.page_content}\n</Document>'
            for doc in results
        ]
    )
    return {"context": [formatted_results]}

def answer_node(state: QuestionState):
    context = state["context"]
    question = state["question"]
    answer = llm.invoke(
        [
            SystemMessage(content=f"Answer the question based on the context: {context}"),
            HumanMessage(content=question)
        ]
    )

    return {"answer": answer}

def ask_question_node(state: QuestionState):
    question = input("Type a question: ")
    return {"question": question}

builder = StateGraph(QuestionState)

builder.add_node('ask', ask_question_node)
builder.add_node('tavily', tavily_search_node)
builder.add_node('wikipedia', wikipedia_search_node)
builder.add_node('answer_question', answer_node)

builder.add_edge(START, 'ask')
builder.add_edge('ask', 'tavily')
builder.add_edge('ask', 'wikipedia')
builder.add_edge(['tavily', 'wikipedia'], 'answer_question')
builder.add_edge('answer_question', END)

graph = builder.compile()

print(graph.get_graph().draw_ascii())

result = graph.invoke({"context": []})
print(result["answer"].content)
