import multiprocessing

from langchain_community.chat_models import ChatLlamaCpp

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
    # verbose=True,
)

messages = [
    ('system', 'you ar a helpful assistant that translates English to Japanese.'),
    ('human', 'I love programming.')
]

ai_msg = llm.invoke(messages)
print(ai_msg)
