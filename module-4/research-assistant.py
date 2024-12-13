from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

import operator
from datetime import datetime
from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, get_buffer_string, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.pregel import RetryPolicy

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create Analysts and review them with human-in-the-loop feedback

class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst")
    name: str = Field(description="Name of the analyst")
    role: str = Field(description="Role of the analyst in the context of the topic")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}"

class Perspectives(BaseModel):
    analysts: list[Analyst] = Field(description="Comprehensive list of analysts with their roles and affiliations.")

class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: list[Analyst]

analyst_instructions="""You are tasted with creating a set of AI analyst personas.
Follow these instructions carefully:
1. First, review the research topic:
`{topic}`

2. Examine any editorial feedback that has been provided to guide the creation of the analysts:
`{human_analyst_feedback}`

3. Determine the most interesting themes based upon documents and / or feedback above.

4. Pict the top {max_analysts} themes.

5. Assign one analyst to each theme.
"""

def create_analysts(state: GenerateAnalystsState):
    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "NONE")

    prompt = analyst_instructions.format(topic=topic,
                                          max_analysts=max_analysts,
                                          human_analyst_feedback=human_analyst_feedback)

    structured_llm = llm.with_structured_output(Perspectives)
    response = structured_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Generate a set of analysts.")
    ])

    return {"analysts": response.analysts}

def human_feedback(state: GenerateAnalystsState):
    """ No-op node that should be interrupted on """
    pass

def should_continue(state: GenerateAnalystsState):
    """ Next node to execute """

    human_feedback = state.get("human_analyst_feedback", None)
    if human_feedback is None:
        return "create_analysts"

    return END

builder = StateGraph(GenerateAnalystsState)
builder.add_node(create_analysts)
builder.add_node(human_feedback)

builder.add_edge(START, create_analysts.__name__)
builder.add_edge(create_analysts.__name__, human_feedback.__name__)
builder.add_conditional_edges(human_feedback.__name__, should_continue, [create_analysts.__name__, END])

memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)
# print(graph.get_graph().draw_ascii())

# topic = input("What is the topic of the research?\n")
# max_analysts = int(input("How many analysts would you like to create?\n"))
# config = {"configurable": {"thread_id": "1"}}

# while True:
#     print("-" * 35 + "Creating Analysts" + "-" * 35)
#     print(f"Topic: {topic}")
#     print(f"Max Analysts: {max_analysts}")
#     print("-" * 100)
#     for event in graph.stream({"topic": topic, "max_analysts": max_analysts}, config, stream_mode="values"):
#         analysts = event.get("analysts", None)
#         if analysts:
#             for analyst in analysts:
#                 print('\n')
#                 print(analyst.persona)
#                 print("-" * 100)

#     repeat = input("Would you like to provide feedback? (Y/n)")
#     if repeat == "n":
#         break

#     feedback = input("What would you like to change?\n")
#     graph.update_state(config, {"human_analyst_feedback": feedback}, as_node="human_feedback")


# print("-" * 100)
# print("Analysts created successfully!")
# print("-" * 100)


# Conduct Interviews
## Generate Questions from the Analysts side

class InterviewState(MessagesState):
    max_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list
    search_query: str

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic.

Your goal is to boil down to interesting and specific insights related to your topic.
1. Interesting: Insights that people will find surprising or non-obvious.
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals:
`{goals}`

Begin by introducing yourself using a name that fits your persona, then ask your question.
Continue to ask questions to drill down and refine your understanding of the topic.
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"
Remember to stay in character throughout your response, reflecting the persona and goals provided to you.
"""

def generate_question(state: InterviewState):
    messages = state["messages"]
    analyst = state["analyst"]

    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([
        SystemMessage(content=system_message),
        *messages
    ])

    return {"messages": [question]}


## Generate Answers from the Expert side

### Generate a search query

search_query_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert.
Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

First, analyze the full conversation.
Pay particular attention to the final question posed by the analyst.
Convert this final question into a well-structured web search query.
If you want to include a specific date, take into account that today's date is {datetime.now().strftime("%Y-%m-%d")}.
""")

tavily_search = TavilySearchResults(max_results=3,topic="news")

def generate_search_query(state: InterviewState):
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([
        search_query_instructions,
        *state["messages"]
    ])

    print(f"Search Query: {search_query.search_query}")
    return {"search_query": search_query.search_query}

### Search in Tavily and Wikipedia
def search_web(state: InterviewState):
    """ Retrieves docs from web search, formats them, and adds them to the context """

    search_query = state["search_query"]
    search_results = tavily_search.invoke(search_query)

    formatted_results = "\n\n---\n\n".join([
        f'<Document href="{result["url"]}"/>\n{result["content"]}\n</Document>'
        for result in search_results
    ])

    return {"context": [formatted_results]}

def search_wikipedia(state: InterviewState):
    """ Retrieves docs from wikipedia, formats them, and adds them to the context """

    search_query = state["search_query"]
    search_results = WikipediaLoader(query=search_query, load_max_docs=3).load()
    formatted_results = "\n\n---\n\n".join([
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
        for doc in search_results
    ])

    return {"context": [formatted_results]}

### Generate an Answer

answer_instructions = """You are an expert being interviewed by an analyst.
Here is the analyst area of focus:
`{goals}`

Your goal is to answer a question posed by the interviewer.
To answer the question, use this context:
```
{context}
```

When answering questions, follow these guidelines:
1. Use only the information provided in the context.
2. Do not introduce external information or make assumptions beyond what's explicitly stated in the context.
3. The context contains source at the topic of each document.
Include these sources in your answer next any relevant states.
For example, for source # 1 use [1].
4. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc..
5. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list:
`[1] assistant/docs/llama3_1.pdf, page 7` and skip the addition of the brackets as well as the `Document` tag.
"""

def generate_answer(state: InterviewState):
    messages = state["messages"]
    context = state["context"]
    analyst = state["analyst"]

    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([
        SystemMessage(content=system_message),
        *messages
    ])

    answer.name = "expert"

    return {"messages": [answer]}

def save_interview(state: InterviewState):
    messages = state["messages"]
    interview = get_buffer_string(messages)

    return {"interview": interview}

def route_messages(state: InterviewState, name: str = "expert"):
    """ Routes between question and answer """

    messages = state["messages"]
    max_turns = state["max_turns"]

    num_turns = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    if num_turns >= max_turns:
        return save_interview.__name__

    # This router runs after each question and answer pair
    # Get the last question asked to check if it signals the end of the interview
    last_question = messages[-2]

    if "Thank you so much for your help!" in last_question.content:
        return save_interview.__name__

    return generate_question.__name__


# Writing the interview down

section_writer_instructions = """You are an expert technical writer.
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents:
- The name of each source document is at the start of the documetn, with the <Document> tag.

2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for the subsection title

3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst:
`{focus}`

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents

6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/

8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""

def write_section(state: InterviewState):
    # interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Use this source to write a section:\n{context}")
    ])

    return {"sections": [section]}

# I'm running into 429 errors when I try to run this graph.
# max_attempts includes the initial attempt. Wait times will be 10, 20, 40 seconds
retry_policy = RetryPolicy(max_attempts=4, initial_interval=10, backoff_factor=2)
interview_builder = StateGraph(InterviewState)
interview_builder.add_node(generate_question, retry=retry_policy)
interview_builder.add_node(generate_search_query, retry=retry_policy)
interview_builder.add_node(search_web, retry=retry_policy)
interview_builder.add_node(search_wikipedia, retry=retry_policy)
interview_builder.add_node(generate_answer, retry=retry_policy)
interview_builder.add_node(save_interview, retry=retry_policy)
interview_builder.add_node(write_section, retry=retry_policy)

interview_builder.add_edge(START, generate_question.__name__)
interview_builder.add_edge(generate_question.__name__, generate_search_query.__name__)
interview_builder.add_edge(generate_search_query.__name__, search_web.__name__)
interview_builder.add_edge(generate_search_query.__name__, search_wikipedia.__name__)
interview_builder.add_edge([search_web.__name__, search_wikipedia.__name__], generate_answer.__name__)
interview_builder.add_conditional_edges(generate_answer.__name__, route_messages, [generate_question.__name__, save_interview.__name__])
interview_builder.add_edge(save_interview.__name__, write_section.__name__)
interview_builder.add_edge(write_section.__name__, END)

# interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")

# print(interview_graph.get_graph().draw_ascii())
#                                   +-----------+
#                                   | __start__ |
#                                   +-----------+
#                                          *
#                                          *
#                                          *
#                                 +-----------------+
#                                 | create_analysts |
#                                 +-----------------+
#                                          *
#                                          *
#                                          *
#                                 +----------------+
#                                 | human_feedback |
#                                 +----------------+
#                                          .
#                                          .
#                                          .
#                               +-------------------+
#                               | conduct_interview |
#                              *+-------------------+***
#                        ******            *            ******
#                   *****                   *                 *****
#                ***                        *                      ***
# +------------------+           +--------------------+           +--------------+
# | write_conclusion |           | write_introduction |           | write_report |
# +------------------+***        +--------------------+       ****+--------------+
#                        ******             *           ******
#                              *****       *       *****
#                                   ***    *    ***
#                                 +-----------------+
#                                 | finalize_report |
#                                 +-----------------+
#                                          *
#                                          *
#                                          *
#                                     +---------+
#                                     | __end__ |
#                                     +---------+


# print('-' * 100)
# print('Conduct Interviews')
# print('-' * 100)

# analyst = graph.get_state(config).values.get("analysts")[0]
# print(analyst.persona)
# print('-' * 100)

# Kick off the interview using the topic
# messages = [HumanMessage(content=f"So, you said you were writing an article on {topic}?")]
# result = interview_graph.invoke({"analyst": analyst, "max_turns": 3, "messages": messages}, config)

# print('-' * 100)
# print('Interview')
# print(result['interview'])
# print('-' * 100)

# print('-' * 100)
# print('Sections')
# print(result['sections'])
# print('-' * 100)

# print('-' * 100)
# print('Context')
# print(result['context'])
# print('-' * 100)


# Finalize the report

class ResearchGraphState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str

def initiate_all_interviews(state: ResearchGraphState):
    """ Conduct an interview with each analyst """

    human_analyst_feedback = state.get("human_analyst_feedback")
    if human_analyst_feedback:
        return create_analysts.__name__

    topic = state["topic"]
    messages = [HumanMessage(content=f"So, you said you were writing an article on {topic}?")]
    # conduct_interview is the name of the node where we assigne the interviewer_graph
    return [Send(
            "conduct_interview",
            {"analyst": analyst,
             "max_turns": 3,
             "messages": messages
            }) for analyst in state["analysts"]]

report_writer_instructions = """You are a technical writer creating a report on this overall topic:

{topic}

You have a team of analysts. Each analyst has done two things:

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task:

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos.
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:

1. Use markdown formatting.
2. Include no pre-amble for the report.
3. Use no sub-heading.
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from:

{context}"""

def write_report(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)
    report = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")])
    return {"content": report.content}

intro_conclusion_instructions = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting.

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header.

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""

def write_introduction(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    intro = llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")])
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Summarize the sections into a final report

    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)
    conclusion = llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")])
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}

# Add nodes and edges
builder = StateGraph(ResearchGraphState)
builder.add_node(create_analysts)
builder.add_node(human_feedback)
builder.add_node("conduct_interview", interview_builder.compile()) # Compile the interview graph (without memory??)
builder.add_node(write_report)
builder.add_node(write_introduction)
builder.add_node(write_conclusion)
builder.add_node(finalize_report)

# Logic
builder.add_edge(START, create_analysts.__name__)
builder.add_edge(create_analysts.__name__, human_feedback.__name__)
builder.add_conditional_edges(human_feedback.__name__, initiate_all_interviews, [create_analysts.__name__, "conduct_interview"])
builder.add_edge("conduct_interview", write_report.__name__)
builder.add_edge("conduct_interview", write_introduction.__name__)
builder.add_edge("conduct_interview", write_conclusion.__name__)
builder.add_edge([
    write_conclusion.__name__,
    write_report.__name__,
    write_introduction.__name__
], finalize_report.__name__)
builder.add_edge(finalize_report.__name__, END)

# Compile
memory = MemorySaver()
graph = builder.compile(interrupt_before=[human_feedback.__name__], checkpointer=memory)

print(graph.get_graph().draw_ascii())
print('-' * 100)

# Run the graph

topic = input("What is the topic of the research?\n")
max_analysts = int(input("How many analysts would you like to create?\n"))
config = {"configurable": {"thread_id": "1"}}

while True:
    print("-" * 35 + "Creating Analysts" + "-" * 35)
    print(f"Topic: {topic}")
    print(f"Max Analysts: {max_analysts}")
    print("-" * 100)
    for event in graph.stream({"topic": topic, "max_analysts": max_analysts}, config, stream_mode="values"):
        analysts = event.get("analysts", None)
        if analysts:
            for analyst in analysts:
                print('\n')
                print(analyst.persona)
                print("-" * 100)

    repeat = input("Would you like to provide feedback? (Y/n)")
    if repeat == "n":
        # Set the human feedback to None so the `initiate_all_interviews` selects the `conduct_interview` nodes
        graph.update_state(config, {"human_analyst_feedback": None}, as_node="human_feedback")
        break

    feedback = input("What would you like to change?\n")
    graph.update_state(config, {"human_analyst_feedback": feedback}, as_node="human_feedback")


print("-" * 100)
print("Analysts created successfully!")
print("-" * 100)
print("Running interviews and writing report...")
print("-" * 100)

# Continue
for event in graph.stream(None, config, stream_mode="updates"):
    print("--Node--")
    node_name = next(iter(event.keys()))
    print(node_name)

final_state = graph.get_state(config)
print("\n" + "-"*100)
print("Final State Values:")
print("-"*100)
for key, value in final_state.values.items():
    print(f"\n{key}:")
    print("-"*50)
    print(value)
    print("-"*100)