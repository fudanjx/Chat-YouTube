from langchain.chat_models import ChatAnthropic
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from streamlit_callback import StreamlitCallbackHandler
import streamlit as st
import os
import anthropic
from langchain.document_loaders import YoutubeLoader

class Text_Expert:
    def __init__(self):
        self.system_prompt = self.get_system_prompt()

        self.user_prompt = HumanMessagePromptTemplate.from_template("{user_question}")

        full_prompt_template = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.user_prompt]
        )

        self.chat = ChatAnthropic(model='claude-v1-100k', max_tokens_to_sample=512, streaming=True, callbacks=[StreamlitCallbackHandler()])

        self.chain = LLMChain(llm=self.chat, prompt=full_prompt_template)

    def get_system_prompt(self):
        system_prompt = """
        You are a expert in script review and summarization

        Please write a professional summary with a beginning, middle, and end based on the video transcript.

        Instructions
        - Do not include any details that are not mentioned in the video.
        - Use bullet points and median section headers to organize your summary.
        - Do not include any personal opinions or thoughts.
        - Include title, introduction, witnesses, their testimonies, key takeaways and conclusion.
        
        Please do not answer anything outside of the context, and be brief.

        ### Video Transcripts 
        {context}
        ### END OF Video Transcripts    
        
        """

        return SystemMessagePromptTemplate.from_template(system_prompt)

    def run_chain(self, language, context, question):
        return self.chain.run(
            language=language, context = context, user_question=question
        )


def load_transcript(url):
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=True,
    )
    docs = loader.load()
    if len(docs) == 0:
        raise ValueError("Sorry, No transcript found.😢")
    return docs[0]

    return text

# create a streamlit app
st.title("Personal YouTube Review Assistant")
st.write("(You may refresh the page to start over)")
anthropic.api_key = st.text_input("###### Enter Anthropic API Key", type="password")
os.environ['ANTHROPIC_API_KEY']= anthropic.api_key

 
url_str = st.text_input("###### Please enter the YouTube url")

if len(url_str) >3: 
    Youtube_transcripts = load_transcript(url=url_str)

    # if an YouTube url is provided
    if Youtube_transcripts:
        st.session_state.context = Youtube_transcripts
    if anthropic.api_key:
        # if there's context_01 & context_02, proceed
        if ("context" in st.session_state):
            # create a text input widget for a question
            question = st.text_input("Ask a question")
        
            # create a button to run the model
            if st.button("Run"):
                # run the model
                tx_expert = Text_Expert()
                bot_response = tx_expert.run_chain(
                    'English', st.session_state.context, 
                        question)

                if "bot_response" not in st.session_state:
                    st.session_state.bot_response = bot_response

                else:
                    st.session_state.bot_response = bot_response

        # display the response
        # if "bot_response" in st.session_state:
            # st.write(st.session_state.bot_response)
    else:
        pass
else:
    pass