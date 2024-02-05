# Required Libraries
import requests
import re
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
import warnings
warnings.filterwarnings("ignore") 

# Function to validate and modify URL
def validate_and_modify_url(url):
    if not url.startswith("https://"):
        if url.startswith("www."):
            url = "https://" + url
        else: url
    return url
# Get URL from the user
user_url = input("Enter the URL of the webpage: ")

# Validate and modify URL
user_url = validate_and_modify_url(user_url)

# Agent 1: Data Scraping and Storage Agent 
response = requests.get(user_url)
raw_content = BeautifulSoup(response.content, 'html.parser').text

# Extract the domain name from the URL
url = response.url
domain = urlparse(url).netloc

# Define the file name using the domain name
store_agent_path ="scrapped_data\\"+ f"{domain}_text.json"

with open(store_agent_path, 'w') as file:
    json.dump({ 'content': raw_content }, file, indent=4)

print(f"Data successfully scraped and stored at {store_agent_path} " )
print()
 
# Agent 2: Query Answering Agent
loader = TextLoader(file_path=store_agent_path, encoding="utf-8")
data = loader.load()
char_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splits = char_text_splitter.split_documents(data)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = 'DB/chromadb'
vector_db = Chroma.from_documents(documents = splits, embedding = embedding_function, persist_directory = persist_directory)
vector_db.persist()

# Load OpenAI
llm = OpenAI(api_key = "YOUR API KEY", temperature=0.7, model_name="gpt-3.5-turbo",  )

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template="""From given list of documents {context} answer the question.You can also use chat history.Construct answer based on the following rules:1. return answer only from the documents. 2. If you don't know the answer, just say that you don't know, don't try to make up an answer. 3. answer the question in  procedural way.{question}""")

# Run chain
qna_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

# Interact with User
while True:  
    question = input("Enter the Question \n") 
    print()  
    result = qna_chain({"query": question})  
    print("Answer : ", result["result"])  
    print()