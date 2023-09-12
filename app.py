import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


# App framework
st.title('TouchGrass.fyi')
prompt = st.text_input('Input your location here to get some suggestions on what you can do outside, NERD')


# Prompt templates
suggestions_template = PromptTemplate(
    input_variables = ['location'],
    template = 'give me suggestions of how I can spend time outdoors, disconnect from technology, and engage with the physical world if I live in {location}'
)

script_template = PromptTemplate(
    input_variables = ['suggestions', 'wikipedia_research'],
    template = 'use a condescending and sassy tone to give me a list of some events happening around me based on these suggestions:{suggestions} while leveraging this wikipedia research:{wikipedia_research}'
)

# Memory 
suggestions_memory = ConversationBufferMemory(input_key= 'location', memory_key= 'chat history')
script_memory = ConversationBufferMemory(input_key= 'suggestions', memory_key= 'chat history')


#Llms
llm = OpenAI(temperature=0.9)
suggestions_chain = LLMChain(llm=llm, prompt = suggestions_template,verbose=True, output_key = 'suggestions', memory=suggestions_memory)
script_chain = LLMChain(llm=llm, prompt = script_template,verbose=True, output_key = 'script', memory=script_memory )
wiki = WikipediaAPIWrapper()

#show stuff to the screen if there's a prompt
if prompt: 
    suggestions = suggestions_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(suggestions=suggestions, wikipedia_research=wiki_research)
    

    if st.button("Touch Grass"): st.write(script)
    
    with st.expander('Suggestions History'):
        st.info(suggestions_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)    

    with st.expander('Wikipedia Research'):
        st.info(wiki_research) 