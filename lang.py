from langchain.document_loaders import UnstructuredPDFLoader
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

loaders= [UnstructuredPDFLoader(os.path.join("data/", fn)) for fn in os.listdir("data")]

docs=[]
for loader in loaders:
    raw_documents = loader.load()
    docs.extend(raw_documents)
text_splitter = CharacterTextSplitter(chunk_size=200,chunk_overlap=50)
splited_docs = text_splitter.split_documents(docs)
db = FAISS.from_documents(splited_docs, OpenAIEmbeddings())

retriever = db.as_retriever()
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                 chain_type="stuff", 
                                 retriever=retriever, 
                                 return_source_documents=True
                                 )

while True:
    inquery = input("入力>> ")
    if inquery == "quit":
        break
    result = qa(inquery)
    print(result["result"])
    print(result["source_documents"])