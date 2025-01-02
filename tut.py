import multiprocessing

from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from pprint import pprint

from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(verbose=True)


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(
    model='qwen2.5:q4',
    temperature=0.1,
    num_thread=multiprocessing.cpu_count() - 1,
    repeat_penalty=1.5,
    top_p=0.9,
    verbose=False,
)

tool = TavilySearchResults(max_results=2)
tools = [tool]

llm_with_tools = llm.bind_tools(
    tools=tools,
    # tool_choice={'type': 'function', 'function': {'name': 'tavily_search_results_json' }},
)

# pprint(llm_with_tools.invoke('名古屋の天気を調べて'))


def chatbot(state: State):
    return {'messages': [llm_with_tools.invoke(state['messages'])]}


graph_builder = StateGraph(State)
graph_builder.add_node('chatbot', chatbot)
graph_builder.add_node('tools', ToolNode(tools=tools))

graph_builder.add_conditional_edges(
    'chatbot',
    tools_condition,
)
graph_builder.add_edge('tools', 'chatbot')
graph_builder.set_entry_point('chatbot')

graph = graph_builder.compile(
    checkpointer=MemorySaver(),
    # interrupt_before=['tools'],
)

graph.get_graph().draw_mermaid_png(output_file_path='./graph.png')
print(graph.get_graph().draw_ascii())

def stream_graph_updates(user_input: str):
    messages = None if user_input == '' else [('user', user_input)]
    config = {'configurable': {'thread_id': '1'}}
    events = graph.stream({'messages': messages}, config, stream_mode='values')
    for event in events:
        if 'messages' in event:
            event['messages'][-1].pretty_print()
    snapshot = graph.get_state(config)
    print(snapshot.next)

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
