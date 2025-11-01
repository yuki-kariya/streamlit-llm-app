import os

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


system_template = """
あなたは医療・ヘルスケア領域の専門家です。
科学的根拠に基づいた説明を行なってください。
診断、処方は決してしないでください。
読み手に安心感を与えるような優しい口調で回答してください。
万が一診察・治療が必要だと判断した場合は、必ず医療機関の受診を促し、あなた自身の判断で診断・処方を行わないでください。
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

healthcare_chain = chat_prompt | llm
