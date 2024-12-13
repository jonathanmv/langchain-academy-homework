from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

import operator
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from langchain_openai import ChatOpenAI


###
# The gist of this file is:
# 1. Generate values dynamically
# 2. Add a conditional edge to "map" to different nodes
# 3. Use operator.add to add values to "reduce" the results
###

subjects_prompt = """Generate a list of 3 sub-topics related to {topic}."""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one!
Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n{jokes}"""


llm = ChatOpenAI(model="gpt-4o", temperature=0.3)


class State(TypedDict):
    topic: str
    subjects: list[str]
    jokes: Annotated[list, operator.add]
    best_joke: str

def get_topic_from_user_node(state):
    topic = input("Enter a topic to generate jokes about: ")
    return {"topic": topic}

def generate_subjects_node(state):
    class Subjects(BaseModel):
        subjects: list[str] = Field(description="A list of 3 sub-topics related to the topic")

    prompt = subjects_prompt.format(topic=state["topic"])
    response = llm.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}

def generate_joke_node(state):
    class Joke(BaseModel):
        joke: str

    prompt = joke_prompt.format(subject=state["subject"])
    joke = llm.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [joke.joke]}

def map_to_generate_joke_node(state):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

def select_best_joke_node(state):
    class BestJoke(BaseModel):
        best_joke: int

    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = llm.with_structured_output(BestJoke).invoke(prompt)
    best_joke = state["jokes"][response.best_joke] if response.best_joke < len(state["jokes"]) else None
    return {"best_joke": best_joke}

builder = StateGraph(State)
builder.add_node("get_topic_from_user", get_topic_from_user_node)
builder.add_node("generate_subjects", generate_subjects_node)
builder.add_node("generate_joke", generate_joke_node)
builder.add_node("select_best_joke", select_best_joke_node)

builder.add_edge(START, "get_topic_from_user")
builder.add_edge("get_topic_from_user", "generate_subjects")
# Don't know why the third argument is an array. I thinks not even needed.
builder.add_conditional_edges("generate_subjects", map_to_generate_joke_node, ["generate_joke"])
builder.add_edge("generate_joke", "select_best_joke")
builder.add_edge("select_best_joke", END)

graph = builder.compile()

# print(graph.get_graph().draw_ascii())
#      +-----------+
#      | __start__ |
#      +-----------+
#             *
#             *
#             *
# +---------------------+
# | get_topic_from_user |
# +---------------------+
#             *
#             *
#             *
#  +-------------------+
#  | generate_subjects |
#  +-------------------+
#             .
#             .
#             .
#    +---------------+
#    | generate_joke |
#    +---------------+
#             *
#             *
#             *
#   +------------------+
#   | select_best_joke |
#   +------------------+
#             *
#             *
#             *
#       +---------+
#       | __end__ |
#       +---------+


for chunk in graph.stream({"topic": ""}):
    print(chunk)

# Sometimes it generated 1 subject and sometimes 3.
# I tried making the prompt more specific, but it didn't help.
# I tried adding a validator, but it didn't help.
# I tried looping but after 3 tries it failed.
# https://smith.langchain.com/o/4e4fd1ec-4840-5eef-bd75-2ab764e2a91e/projects/p/a219866a-d579-4129-a0a6-f2fccda487e9?columnVisibilityModel=%7B%22outputs%22%3Afalse%2C%22feedback_stats%22%3Afalse%2C%22reference_example%22%3Afalse%7D&timeModel=%7B%22duration%22%3A%227d%22%7D&runtab=0&peek=37bec397-5dde-45a9-b50d-54f69c1cafd5
