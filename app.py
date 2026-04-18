import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent,Tool
from langchain.callbacks import StreamlitCallbackHandler

#streamlit app
st.set_page_config(page_title="text to math problem solver and data search assistant", page_icon=":robot_face:")
st.title("text to math problem solver and data search assistant")

groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key in the sidebar to use the app.")
    st.stop()
    
llm=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)

#Intialize the Wikipedia API wrapper
wiki = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wiki.run,
    description="A tool for searching the internet to find the various information on the topics mentioned."
    
)

#initial math tool
def calculator_tool(input_text):
    try:
        result = eval(input_text)
        return f"Final Answer: {result}"
    except:
        return "Invalid math expression"

math_tool = Tool(
    name="Calculator",
    func=calculator_tool,
    description="A tool for solving math problems. Input should be a math problem in text format."
)

prompt = """You are a helpful assistant.

IMPORTANT RULES:
- Do NOT generate Python code
- Use tools when needed
- Always respond in plain text
Use Calculator only once for math
- Do NOT repeat tool calls
- Do NOT generate Python code
- After getting result, STOP and give FINAL ANSWER

Question: {question}
Answer:"""


prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)


# #math probleem tool
# chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_chain = prompt_template | llm

def reasoning_func(q):
    return reasoning_chain.invoke({"question": q})


reasoing_tool = Tool(
    name="Reasoning",
    func=reasoning_func,
    description="A tool for logic-based and reasoning questions."
)

agent = initialize_agent(
    tools=[wikipedia_tool, math_tool, reasoing_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    early_stopping_method="generate"
    
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I am your assistant. I can help you solve math problems and search for information on the internet. Please ask me anything!"}
        
    ]  
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    

question = st.chat_input("Ask me anything...")


if question:
    with st.spinner("Generating response..."):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = agent.run(question, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
       

            