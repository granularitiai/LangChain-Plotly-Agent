#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install --upgrade langchain-experimental langgraph langchainhub plotly')


# In[11]:


from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
import os
from langchain_openai import ChatOpenAI
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import getpass


# In[12]:


import getpass


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")



_set_env("OPENAI_API_KEY")


# In[13]:


df = pd.read_csv("Downloads/colorectal_cancer_prediction.csv")


# In[14]:


llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")


# In[15]:


agent_exec = create_python_agent(
    llm = llm,
    tool = PythonREPLTool(),
    verbose = True,
    handle_parsing_errors=True,
)


# In[20]:


agent_exec.run(f""" 
            You are a skilled data visualization engineer. Given the {df}, generate visuals using plotly for the data provided.""")


# In[ ]:




