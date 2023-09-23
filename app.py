import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ["GOOGLE_CSE_ID"] = st.secrets['GOOGLE_CSE_ID']
os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_API_KEY']

# App framework
st.title('TouchGrass.fyi')
prompt = st.text_input('Input your location here to get some suggestions on what you can do outside')
categories = ["Sports", "Arts & Theatre", "Family", "Nature"]
selected_categories = st.multiselect("Select categories of activities you're interested in:", categories)
touch_grass_button = st.button('Touch Grass')

# Prompt templates
suggestions_template = PromptTemplate(
    input_variables=['location', 'selected_categories'],
    template='Given that I\'m interested in {selected_categories}, provide one suggestion on how I can spend time outdoors if I live in {location}'
)

# Memory 
suggestions_memory = ConversationBufferMemory(input_key='location', memory_key='chat history')

# Llms
llm = OpenAI(temperature=0.9)
suggestions_chain = LLMChain(llm=llm, prompt=suggestions_template, verbose=True, output_key='suggestions', memory=suggestions_memory)
search = GoogleSearchAPIWrapper()

def google_search_for_suggestion(suggestion):
    results = search.results(suggestion, 1)  # Get top result
    return results[0]['Link'] if results and results[0].get('Link') else None

if touch_grass_button:
    if prompt and selected_categories:
        formatted_categories = ', '.join(selected_categories)
        suggestion = suggestions_chain.run(location=prompt, selected_categories=formatted_categories)

        # If the model returns multiple suggestions (in a list), take only the first one
        if isinstance(suggestion, list):
            suggestion = suggestion[0]

        google_link = google_search_for_suggestion(suggestion)

        st.write(suggestion)  # Display the suggestion
        if google_link:
            st.markdown(f"[Link to top search result]({google_link})")  # Display the link as a clickable hyperlink

        # Display other info
        with st.expander('Suggestions History'):
            st.info(suggestions_memory.buffer)
