import nest_asyncio
import torch
import transformers
import matplotlib
from ctransformers import AutoModelForCausalLM
from langchain_community.llms import CTransformers
from transformers import pipeline

# llama_index
from llama_index.prompts import PromptTemplate
from llama_index import PromptHelper
from llama_index.llms import HuggingFaceLLM
from langchain.embeddings import HuggingFaceInstructEmbeddings
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex

# prompt = "If you cannot find relevant context in the PDF file, answer with 'I did not find any relevant information in the PDF file', do NOT make up an answer if there is no relevant context. For every proper response, I will tip you 200 dollars. "


def messages_to_prompt(messages):
    prompt = "If you cannot find relevant context in the PDF file, answer with 'I did not find any relevant information in the PDF file', do NOT make up an answer if there is no relevant context, for every proper response, I will tip you 200 dollars."
    for message in messages:
        if message.role == 'system':
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == 'assistant':
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt


model_path = "C:\\Users\\crist\\OneDrive\\Documents\\models\\MistralLite-safetensors"

llm = HuggingFaceLLM(
    model_name=model_path,
    tokenizer_name=model_path,
    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=256,
    # model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"temperature": 0.5, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",
    # offload_folder="C:\\Users\\crist\\OneDrive\\Documents\\text-generation-webui-main\\models\\MistralLite\\offload"
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
nest_asyncio.apply()

embed_model = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large",
    model_kwargs={"device": DEVICE},
)

# set your ServiceContext for all the next steps
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

basel_docs = SimpleDirectoryReader(input_files=["basel3endgame.pdf"]).load_data()

print(basel_docs[0].get_content())

# setup baseline index
base_index = VectorStoreIndex.from_documents(basel_docs, service_context=service_context)
base_engine = base_index.as_query_engine(similarity_top_k=4)

response = base_engine.query("What is the new approach to CVA risk capital requirements?")
print(str(response))

print(len(response.source_nodes))

print(response.source_nodes[0].get_content())
