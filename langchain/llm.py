# ref: https://www.youtube.com/watch?v=ekpnVh-l3YA&list=PL4HikwTaYE0GEs7lvlYJQcvKhq0QZGRVn&index=2&ab_channel=LeonvanZyl
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# Create a new instance of the ChatOpenAI class
# not recommended to put the API key in the code!!!!!!!!!
# llm = ChatOpenAI(
#     api_key="sk-1234567890abcdef1234567890abcdef"
# )
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=1, # 0~1, 0:strict and factual, 1:creative 
    max_tokens=100,
    verbose=True
)

response = llm.invoke("Hello, how are you?")
print(response)

# give a list of strings to the batch method
response = llm.batch(["Hello, how are you?", "What is the meaning of life?"])
print(response)

# stream method is used to get a stream of responses
response = llm.stream("Write a poem about AI")

for chunk in response:
    print(chunk.content, end="", flush=True)
