from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embedings
from langchain_pinecone import PineconeVectorStore
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

embeddings = download_huggingface_embedings()

index_name = "copyia"

docSearch = PineconeVectorStore.from_existing_index(
  index_name=index_name,
  embedding=embeddings
)

retriever = docSearch.as_retriever(
  search_type="similarity",
  search_kwargs={"k": 3}
)

model_name = "facebook/bart-large"
hf_pipeline = pipeline("text2text-generation", model=model_name)

llm = HuggingFacePipeline(pipeline = hf_pipeline)

prompt = ChatPromptTemplate.from_messages( 
  [ 
    ("system", chat_prompt),
    ("human", "{input}"),
  ]
)

question_answering_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)

@app.route("/")
def index():
  return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
  msg = request.form["msg"]
  input = msg
  print(input)
  response = rag_chain.invoke({"input": msg})
  print("Response : ", response["answer"])
  return str(response["answer"])

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8086, debug=True)