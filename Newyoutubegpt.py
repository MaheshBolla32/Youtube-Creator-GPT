# Import necessary libraries and modules
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# Collect OpenAI API key from the user via Streamlit sidebar
api_key = st.sidebar.text_input(
    label="#### Your OpenAI API key 👇",  # Label for the input field
    placeholder="Paste your OpenAI API key, sk-",  # Placeholder text to guide the user
    type="password"  # Input type set to password for secure entry
)

# Create a button in the sidebar for the user to submit the API key
st.sidebar.button('Enter', type="primary")

# Create an OpenAI language model instance with optional API key and temperature setting
llm = OpenAI(api_key=api_key, temperature=0.9) if api_key else None

# Set the title of the Streamlit application
st.title('🦜🔗 YouTube Creator GPT ')

# Create a text input field for the user to input their prompt
prompt = st.text_input('Plug in your prompt here')

# Define a prompt template for generating YouTube video titles
title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a youtube video title about {topic}'
)

# Define a prompt template for generating YouTube video scripts
script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='Write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia research:{wikipedia_research} '
)

# Define a conversation buffer memory for storing YouTube video titles
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

# Define a conversation buffer memory for storing YouTube video scripts
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Create a language model chain for generating YouTube video titles
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory) if llm else None

# Create a language model chain for generating YouTube video scripts
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory) if llm else None

# Instantiate a wrapper for accessing Wikipedia API
wiki = WikipediaAPIWrapper()

# Check if both a prompt and an OpenAI language model instance are provided
if prompt and llm:
    # Generate a YouTube video title using the title_chain if available
    title = title_chain.run(prompt) if title_chain else None

    # Retrieve Wikipedia research information using the WikipediaAPIWrapper
    wiki_research = wiki.run(prompt)

    # Generate a YouTube video script using the script_chain if available
    script = script_chain.run(title=title, wikipedia_research=wiki_research) if script_chain else None

    # Check if a title is generated and display it using Streamlit
    if title:
        st.write(title)

    # Check if a script is generated and display it using Streamlit
    if script:
        st.write(script)

    # Display the title history using a Streamlit expander
    with st.expander('Title History'):
        st.info(title_memory.buffer)  # Display information stored in the title_memory buffer

    # Display the script history using a Streamlit expander
    with st.expander('Script History'):
        st.info(script_memory.buffer)  # Display information stored in the script_memory buffer

    # Display Wikipedia research information using a Streamlit expander
    with st.expander('Wikipedia Research'):
        st.info(wiki_research)  # Display information retrieved from Wikipedia research

# Display a message if the user tries to search without entering an API key
if not api_key and prompt:
    st.warning("Please enter your OpenAI API key before searching.")

st.sidebar.markdown('Made with ❤️ by [Mahesh Bolla](https://www.linkedin.com/in/ubolla/)')
