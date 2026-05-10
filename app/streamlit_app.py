import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_chatbot.config import ModelConfig
from llm_chatbot.gpu import collect_gpu_info
from llm_chatbot.model import ChatModel


st.set_page_config(page_title="L4 LLM Chatbot", layout="wide")


@st.cache_resource(show_spinner="Loading model on the inference device...")
def load_chat_model(model_id: str, use_4bit: bool, max_new_tokens: int, temperature: float, top_p: float) -> ChatModel:
    base = ModelConfig.from_env()
    config = ModelConfig(
        model_id=model_id,
        hf_token=base.hf_token,
        use_4bit=use_4bit,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        require_cuda=base.require_cuda,
    )
    return ChatModel(config)


def sidebar_config() -> tuple[str, bool, int, float, float]:
    default = ModelConfig.from_env()
    with st.sidebar:
        st.header("Model")
        model_id = st.text_input("Hugging Face model ID", value=default.model_id)
        use_4bit = st.toggle("4-bit quantization", value=default.use_4bit)
        max_new_tokens = st.slider("Max new tokens", min_value=32, max_value=1024, value=default.max_new_tokens, step=32)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=default.temperature, step=0.05)
        top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=default.top_p, step=0.05)

        st.header("GPU")
        st.json(collect_gpu_info())

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return model_id, use_4bit, max_new_tokens, temperature, top_p


def main() -> None:
    st.title("L4 LLM Chatbot")
    model_id, use_4bit, max_new_tokens, temperature, top_p = sidebar_config()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        chat_model = load_chat_model(model_id, use_4bit, max_new_tokens, temperature, top_p)
    except Exception as exc:
        st.error("Model could not be loaded. Check MODEL_ID, HF_TOKEN, CUDA, and accepted model licenses.")
        st.exception(exc)
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask the model something...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = ""
        try:
            for chunk in chat_model.stream(st.session_state.messages):
                response += chunk
                placeholder.markdown(response)
        except Exception as exc:
            st.error("Generation failed.")
            st.exception(exc)
            return

    st.session_state.messages.append({"role": "assistant", "content": response.strip()})


if __name__ == "__main__":
    main()
