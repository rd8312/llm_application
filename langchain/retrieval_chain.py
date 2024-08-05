# ref: https://www.youtube.com/watch?v=-Ueh5XBpcoY&list=PL4HikwTaYE0GEs7lvlYJQcvKhq0QZGRVn&index=4&ab_channel=LeonvanZyl
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

docA = Document(
    page_content="LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together.",
)

# Instantiate Model
llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.4,
)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the user's question.
    Context: {context}
    Question: {input}
    """
)

# chain = prompt | llm
chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
response = chain.invoke({"input": "What is LCEL?", "context": [docA]})
print(response)
# output:
# LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. It was designed to
# be a simple and intuitive way to create complex chains of operations, without having to write a lot of code.
# LCEL is based on a few simple concepts, such as expressions, functions, and variables. Expressions are the building blocks of LCEL, 
# and they can be combined in various ways to create more complex expressions. Functions are used to perform operations on data, 
# and variables are used to store and manipulate data. Together, these concepts make it easy to create powerful and flexible chains of operations in LCEL.


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs


print(get_documents_from_web("https://python.langchain.com/docs/expression_language/"))
# [Document(page_content='LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. It was designed to 
# be a simple and intuitive way to create complex chains of operations, without having to write a lot of code. 
# LCEL is based on a few simple concepts, such as expressions, functions, and variables. Expressions are the building blocks of LCEL, 
# and they can be combined in various ways to create more complex expressions. Functions are used to perform operations on data, 
# and variables are used to store and manipulate data. Together, these concepts make it easy to create powerful and flexible chains of operations in LCEL.',
# metadata={'source': 'https://python.langchain.com/docs/expression_language/', 'title': 'LangChain Expression Language', 
# 'description': 'LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. 
# It was designed to be a simple and intuitive way to create complex chains of operations, without having to write a lot of code. 
# LCEL is based on a few simple concepts, such as expressions, functions, and variables. Expressions are the building blocks of LCEL, 
# and they can be combined in various ways to create more complex expressions. Functions are used to perform operations on data, 
# and variables are used to store and manipulate data. Together, these concepts make it easy to create powerful and flexible chains of operations in LCEL.',
# 'language': 'en'})]
docs = get_documents_from_web("https://python.langchain.com/docs/expression_language/")

llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.4,
)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the user's question.
    Context: {context}
    Question: {input}
    """
)

# chain = prompt | llm
chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
response = chain.invoke({"input": "What is LCEL?", "context": docs})
print(response)




# Retrieve Data
def get_docs():
    loader = WebBaseLoader("https://python.langchain.com/docs/expression_language/")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splitDocs = text_splitter.split_documents(docs)
    print(splitDocs) # len(splitDocs) = 22
    return splitDocs


def create_vector_store(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore


def create_chain(vectorStore):
    model = ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo-1106")

    prompt = ChatPromptTemplate.from_template(
        """
    Answer the user's question.
    Context: {context}
    Question: {input}
    """
    )

    # chain = prompt | model
    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)

    retriever = vectorStore.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


docs = get_docs()
vectorStore = create_vector_store(docs)
chain = create_chain(vectorStore)

response = chain.invoke(
    {
        "input": "What is LCEL?",
    }
)

print(response)
