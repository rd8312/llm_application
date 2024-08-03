# ref: https://www.youtube.com/watch?v=hVs8MVydN3A&list=PL4HikwTaYE0GEs7lvlYJQcvKhq0QZGRVn&index=3&ab_channel=LeonvanZyl
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
print(response)

# Prompt Template for Chat
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma seperated list."),
        ("human", "{input}")
    ]
)

# Create LLM Chain
chain = prompt | llm

response = chain.invoke({"input": "happy"})
print(response)
# output: joyful, cheerful, pleased, delighted, content, contented, satisfied, glad, elated, overjoyed
