from dotenv import load_dotenv
import streamlit as st
import os
from enum import Enum
from langchain.schema import AIMessage

load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]


class Domain(Enum):
    HEALTHCARE = "healthcare"
    DX = "dx"


def on_send(domain: Domain, input: str) -> None:
    print(f"Domain: {domain}, Input: {input}")

    res: AIMessage | None = None

    if domain == Domain.HEALTHCARE:
        from src.health_llm_chain import healthcare_chain

        chain = healthcare_chain
        res = chain.invoke({"input": input})
    else:
        from src.dx_llm_chain import dx_chain

        chain = dx_chain
        res = chain.invoke({"input": input})

    st.write(res.content)
    print(f"Response: {res.content}")


def render():
    st.title("専門家 AIチェット🤖")

    st.write(
        """
        <p>こんにちは！</p>
        <p>私は様々な専門分野に関する相談に自動で応答するAIボットです</p>
        <br/>
        <p>気になる専門分野を選択して相談を開始しよう！</p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    domain = st.radio(
        label="専門分野を選択してください",
        options=["医療・ヘルスケア", "ITテクノロジー"],
    )

    domain_value = Domain.HEALTHCARE if domain == "医療・ヘルスケア" else Domain.DX

    q = st.text_area(label="質問内容を入力してください", disabled=not bool(domain))

    is_send = st.button(
        label="送信",
        disabled=len(q) == 0,
        type="primary",
    )

    if is_send:
        on_send(domain=domain_value, input=q)


render()
