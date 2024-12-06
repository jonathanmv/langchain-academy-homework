from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from operator import add
from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]

# One subgraph
class QuestionSummarizationState(TypedDict):
    cleaned_logs: list[Log]
    qs_summary: str
    report: str
    processed_logs: List[str]

class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]

def qs_generate_summary(state):
    cleaned_logs = state["cleaned_logs"]
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return { "qs_summary": summary, "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs]}

def send_to_slack(state):
    qs_summary = state["qs_summary"]
    report = "some report"
    return { "report": report }

qs_builder = StateGraph(input=QuestionSummarizationState, output=QuestionSummarizationOutputState)
qs_builder.add_node("generate_summary", qs_generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)

qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", END)

qs_graph = qs_builder.compile()
print("Question Summarization Subgraph")
print(qs_graph.get_graph().draw_ascii())

# Another subgraph
class FailureAnalysisState(TypedDict):
    cleaned_logs: list[Log]
    failures: List[Log]
    fa_summary: str
    processed_logs: List[str]

class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]

def get_failures(state):
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return { "failures": failures }

def fa_generate_summary(state):
    failures = state["failures"]
    summary = "Poor quality retrieval of Chroma documentation."
    return { "fa_summary": summary, "processed_logs": [f"failure-analysis-on-log-{log['id']}" for log in failures]}

fa_builder = StateGraph(input=FailureAnalysisState, output=FailureAnalysisOutputState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", fa_generate_summary)

fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)

fa_graph = fa_builder.compile()
print("\nFailure Analysis Subgraph")
print(fa_graph.get_graph().draw_ascii())

# Parent Graph
class ParentGraphState(TypedDict):
    raw_logs: list[Log]
    cleaned_logs: list[Log]
    fa_summary: str
    report: str
    processed_logs: Annotated[List[str], add]

def clean_logs(state):
    raw_logs = state["raw_logs"]
    cleaned_logs = raw_logs
    return { "cleaned_logs": cleaned_logs }

builder = StateGraph(ParentGraphState)
builder.add_node("clean_logs", clean_logs)
builder.add_node("question_summarization", qs_graph)
builder.add_node("failure_analysis", fa_graph)

builder.add_edge(START, "clean_logs")
builder.add_edge("clean_logs", "question_summarization")
builder.add_edge("clean_logs", "failure_analysis")
builder.add_edge("question_summarization", END)
builder.add_edge("failure_analysis", END)


graph = builder.compile()
print("\nParent Graph")
print(graph.get_graph().draw_ascii())


# Run the graph
# Dummy logs
question_answer = Log(
    id="1",
    question="How can I import ChatOllama?",
    answer="To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'",
)

question_answer_feedback = Log(
    id="2",
    question="How can I use Chroma vector store?",
    answer="To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
    grade=0,
    grader="Document Relevance Recall",
    feedback="The retrieved documents discuss vector stores in general, but not Chroma specifically",
)

raw_logs = [question_answer,question_answer_feedback]
result = graph.invoke({"raw_logs": raw_logs})
print("\nReport:")
print(result["report"])

print("\nFailure Analysis Summary:")
print(result["fa_summary"])

print("\nProcessed Logs:")
for log in result["processed_logs"]:
    print(log)
