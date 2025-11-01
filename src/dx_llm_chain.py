import os

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

system_template = """
あなたはDX領域の専門家です。
IT技術に詳しくないユーザーが質問することもなるので、専門用語の説明を適宜行ない、回答してください。
"""
system_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_prompt = HumanMessagePromptTemplate.from_template("{input}")

chat_prompt = ChatPromptTemplate.from_messages(
    [
        system_prompt,
        human_prompt,
    ]
)

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

dx_chain = chat_prompt | llm
