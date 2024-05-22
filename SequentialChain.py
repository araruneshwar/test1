from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SimpleSequentialChain
import streamlit as st

st.title("cuisine food plan generator")

cuisine = st.text_input("enter the cuisine name")

cuisineFood_template = PromptTemplate(
    input_variables = ['cuisine'],
    template = "Human: Give me {cuisine} cuisine based healthy food dish for breakfast, lunch and dinner."    
)

foodRecipe_template = PromptTemplate(
    input_variables = ['dish'],
    template = "Human: Generate food receipe for {dish} in a simple step by step instruction for one serving along with the calories, micro nutrients and macro nutrients in a table format."    
)
foodCalorie_template = PromptTemplate(
    input_variables = ['food'],
    template = "Human: give the calories, micro nutrients and macro nutrients that are present in {food}. Give output in a table format."    
)

llm = ChatOpenAI(openai_api_base="http://localhost:1234/v1",max_tokens=3000)
cuisineChain = LLMChain(llm=llm,prompt=cuisineFood_template)
RecipeChain = LLMChain(llm=llm,prompt=foodRecipe_template)
foodChain = LLMChain(llm=llm,prompt=foodCalorie_template)

OverAllChain = SimpleSequentialChain(chains=[cuisineChain,foodChain,RecipeChain],verbose=True,strip_outputs=True)

if cuisine:
    # response = llm(foodCalorie_template.format(food=food))
    response = OverAllChain.run(cuisine)

    st.write(response)