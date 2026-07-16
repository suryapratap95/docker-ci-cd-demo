import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS


from langchain_core.runnables import RunnablePassthrough

load_dotenv()
INDEX_DIR = "faiss_index"

SYSTEM_PROMPT = """ you are a helpful chat assitant for ABC Bank,
Answer users question only from the provided contet
Rules:
 - always try to give answer in empathtic and polite tone
- if you don't know answer please say i don't have information
- Keep answer short, clear and to the point
- don't use complex banking terms answer in plain english 
- never invent interst rates, policies and fees , if you don't have inormation say i  do not know

"""

def format_docs(docs) -> str:
    """Turn a list of retreieved doccuments into a single text block for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)

def load_chain():

    if  not os.path.isdir(INDEX_DIR):
        raise FileNotFoundError(
            f" '{INDEX_DIR}' not found. Run 'python ingest.py ' to build the vector knowledge base"
        )
        
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True 
    )
    retreiver = vector_store.as_retriever(search_kwargs = {"k": 3})
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context: \n {context} \n\n customer question : {question}")
        ]
    )
    
    

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1
    )

    chain = (
        {"context": retreiver | format_docs, "question": RunnablePassthrough()}
        | prompt 
        | llm
        | StrOutputParser()
    )
    
    return chain

