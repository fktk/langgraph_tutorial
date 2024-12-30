import os
import json
from dotenv import load_dotenv
from typing import Annotated
from pprint import pprint

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage


load_dotenv(verbose=True)
TAVILY = os.environ.get('TAVILY_API_KEY')


llm = ChatOpenAI(
    base_url='http://localhost:8080',
    api_key='hoge',
)

tool = TavilySearchResults(max_results=2)
print(tool.name)
tools = [tool]

llm_with_tools = llm.bind_tools(
    tools=tools,
    tool_choice={'type': 'function', 'function': {'name': 'tavily_search_results_json' }},
)
print(llm_with_tools.invoke('hello'))


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {'messages': [llm_with_tools.invoke(state['messages'])]}


class BasicToolNode:

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get('messages', []):
            message = messages[-1]
        else:
            raise ValueError('No message found in input')
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call['name']].invoke(
                tool_call['args']
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call['name'],
                    tool_call_id=tool_call['id'],
                )
            )
        return {'messages': outputs}


def route_tools(state: State):

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get('messages', []):
        ai_message = messages[-1]
    else:
        raise ValueError(f'No messages found in input state to tool_edge: {state}')
    if hasattr(ai_message, 'tool_calls') and len(ai_message.tool_calls) > 0:
        return 'tools'
    return END


tool_node = BasicToolNode(tools=tools)

graph_builder = StateGraph(State)
graph_builder.add_node('chatbot', chatbot)
graph_builder.add_node('tools', tool_node)

graph_builder.add_edge(START, 'chatbot')
# graph_builder.add_edge('chatbot', END)
graph_builder.add_conditional_edges(
    'chatbot',
    route_tools,
    {'tools': 'tools', END: END}
)
graph_builder.add_edge('tools', 'chatbot')

graph = graph_builder.compile()

# graph.get_graph().draw_mermaid_png(output_file_path='./graph.png')
print(graph.get_graph().draw_ascii())

def stream_graph_updates(user_input: str):
    for event in graph.stream({'messages': [('user', user_input)]}):
        for value in event.values():
            print('Assistant:', value['messages'][-1].content)

# while True:
#     try:
#         user_input = input('User: ')
#         if user_input.lower() in ['quit', 'exit', 'q']:
#             print('Goodbye!')
#             break
#
#         stream_graph_updates(user_input)
#     except:
#         user_input = 'What do you know about LangGraph?'
#         print('User: ' + user_input)
#         stream_graph_updates(user_input)
#         break
