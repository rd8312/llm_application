# ref: https://www.youtube.com/watch?v=qFGdygmYgto&list=PL4HikwTaYE0GEs7lvlYJQcvKhq0QZGRVn&index=3&ab_channel=LeonvanZyl
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Instantiate Model
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-3.5-turbo-1106",
)

# Prompt Template
prompt = ChatPromptTemplate.from_template("Tell me a joke about a {subject}.")

# Create LLM Chain
chain = prompt | llm

response = chain.invoke({"subject": "dog"})
print(type(response))
# output: <class 'langchain_core.messages.ai.AIMessage'>

response = chain.invoke({"subject": "dog"})
print(type(response.content))
# output: <class 'str'>

def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tell me a joke about the following subject"),
        ("human", "{input}")
    ])

    parser = StrOutputParser()

    chain = prompt | llm | parser

    return chain.invoke({
        "input": "dog"
    })

def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma seperated list."),
        ("human", "{input}")
    ])

    parser = CommaSeparatedListOutputParser()
    
    chain = prompt | llm | parser

    return chain.invoke({
        "input": "happy"
    })

def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract information from the following phrase.\nFormatting Instructions: {format_instructions}"),
        ("human", "{phrase}")
    ])

    class Food(BaseModel):
        recipe: str = Field(description="the name of the recipe")
        ingredients: list = Field(description="ingredients")       

    parser = JsonOutputParser(pydantic_object=Food)

    chain = prompt | llm | parser
    
    return chain.invoke({
        "phrase": "The ingredients for a Margherita pizza are tomatoes, onions, cheese, basil",
        "format_instructions": parser.get_format_instructions()
    })

print(type(call_string_output_parser()))
# output: <class 'str'>

print(type(call_list_output_parser()))
# output: [joyful, cheerful, pleased, delighted, content, contented, satisfied, glad, elated, overjoyed]
print(type(call_list_output_parser()))
# output: <class 'list'>

print(call_json_output_parser())
# output: {'recipe': 'Margherita pizza', 'ingredients': ['tomatoes', 'onions', 'cheese', 'basil']}

