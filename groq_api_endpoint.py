from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader

import time
start_time = time.time()


load_dotenv() 
groq_api_key = os.environ['GROQ_API_KEY']

llm_groq = ChatGroq(
            #groq_api_key=groq_api_key,
            # model_name='llama3-8b-8192' 
            model_name='mixtral-8x7b-32768'
    )


files= SimpleDirectoryReader("/home/sid/Documents/my rag chatbot/data").load_data()
pdf_text = " ".join(map(str, files))
        

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(pdf_text)

# Create a metadata for each chunk
metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

# Create a Chroma vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory="/home/sid/Documents/my rag chatbot/vectorDB")
# docsearch.persist()

docsearch = Chroma(persist_directory="/home/sid/Documents/my rag chatbot/vectorDB", embedding_function=embeddings)

print("--- %s seconds ---" % (time.time() - start_time))

message_history = ChatMessageHistory()
    
    # Memory for conversational context
memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )


from langchain.prompts import PromptTemplate

# Define your system instruction
system_instruction = "The assistant should provide detailed explanations and act as pet expert and give answers in a friendly manner."

# Define your template with the system instruction
template = (
    f"{system_instruction} "
    "Combine the chat history and follow up question into "
    "a standalone question. Chat History: {chat_history}"
    "Follow up question: {question}"
)

# Create the prompt template
condense_question_prompt = PromptTemplate.from_template(template)

# Now you can pass this prompt to the from_llm method
chain = ConversationalRetrievalChain.from_llm(
    llm=llm_groq,
    retriever=docsearch.as_retriever(search_kwargs={"k": 4}),
    condense_question_prompt=condense_question_prompt,
    chain_type="stuff",
    memory=memory,
)




def query(user_question):
    
        response = chain.invoke(user_question)
        answer = response["answer"]
        return answer


from fastapi import FastAPI
import nest_asyncio
from pyngrok import ngrok
import uvicorn

app = FastAPI()

@app.get("/query")
def query_response(ques: str):
        answer = query(ques)
        return {"question": ques, "answer": answer}

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
