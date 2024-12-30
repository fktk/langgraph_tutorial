import os
import json
import multiprocessing

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from pprint import pprint

from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(verbose=True)


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatLlamaCpp(
    temperature=0.1,
    model_path='./models/qwen2.5-3b-instruct-q4_k_m.gguf',
    n_ctx=10000,
    n_gpu_layers=20,
    n_batch=300,
    max_tokens=512,
    n_threads=multiprocessing.cpu_count() - 1,
    repeat_penalty=1.5,
    top_p=0.9,
    verbose=False,
)

pprint('start-----------')

tool = TavilySearchResults(max_results=2)
tools = [tool]

llm_with_tools = llm.bind_tools(
    tools=tools,
    tool_choice={'type': 'function', 'function': {'name': 'tavily_search_results_json' }},
)


def chatbot(state: State):
    return {'messages': [llm.invoke(state['messages'])]}


graph_builder = StateGraph(State)
graph_builder.add_node('chatbot', chatbot)
graph_builder.add_node('tools', ToolNode(tools=tools))

graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition,
)
graph_builder.add_edge('tools', 'chatbot')
graph_builder.set_entry_point('chatbot')

graph = graph_builder.compile(checkpointer=MemorySaver())

# graph.get_graph().draw_mermaid_png(output_file_path='./graph.png')
print(graph.get_graph().draw_ascii())

def stream_graph_updates(user_input: str):
    config = {'configurable': {'thread_id': '1'}}
    for event in graph.stream({'messages': [('user', user_input)]}, config):
        for value in event.values():
            print('Assistant:', value['messages'][-1].content)

while True:
    try:
        user_input = input('User: ')
        if user_input.lower() in ['quit', 'exit', 'q']:
            print('Goodbye!')
            break

        stream_graph_updates(user_input)
    except:
        user_input = 'What do you know about LangGraph?'
        print('User: ' + user_input)
        stream_graph_updates(user_input)
        break
