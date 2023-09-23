import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
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
custom_category = st.text_input("Enter your own interest if it's not listed above:")

if custom_category:
    selected_categories.append(custom_category)

touch_grass_button = st.button('Touch Grass 👉🌿')

# Prompt templates
suggestions_template = PromptTemplate(
    input_variables=['location', 'selected_categories'],
    template='Given that Im interested in {selected_categories}, provide suggestions on how I can spend time outdoors if I live in {location}'
)

script_template = PromptTemplate(
    input_variables=['suggestions', 'wikipedia_research'],
    template='give me an idea of some location specific activities near me based on these suggestions:{suggestions} while leveraging this wikipedia research:{wikipedia_research}'
)

# Memory 
suggestions_memory = ConversationBufferMemory(input_key= 'location', memory_key= 'chat history')
script_memory = ConversationBufferMemory(input_key= 'suggestions', memory_key= 'chat history')

# Llms
llm = OpenAI(temperature=0.9)
suggestions_chain = LLMChain(llm=llm, prompt=suggestions_template, verbose=True, output_key='suggestions', memory=suggestions_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
wiki = WikipediaAPIWrapper()
search = GoogleSearchAPIWrapper(k=1)


# Extract individual suggestions from the numbered list
def extract_suggestions_from_list(suggestions_text):
    suggestions = suggestions_text.split("\n")
    # Remove numbering and strip any whitespace
    return [suggestion.split(" ", 1)[1].strip() for suggestion in suggestions if suggestion]


# Execute when button is clicked
if touch_grass_button:
    if prompt and selected_categories:
        formatted_categories = ', '.join(selected_categories)
        suggestions_text = suggestions_chain.run(location=prompt, selected_categories=formatted_categories)
        
        suggestion_list = extract_suggestions_from_list(suggestions_text)
        
        for suggestion in suggestion_list:
            # Make an individual call to the API for each suggestion
            google_result = search.run(suggestion)
    
    # Ensure the result contains the 'link' key
    link = google_result.get('link')
    if link:
        st.write(f"**{suggestion}** - [Link]({link})")
    else:
        st.write(f"**{suggestion}** - No top link found")
        
       
