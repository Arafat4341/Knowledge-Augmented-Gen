# Importing required libraries
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
import wget

from dotenv import load_dotenv
import os

# ============== Indexing =================
"""
Loading
Load the document
The document, which is provided in a TXT format,
outlines some company policies and serves as an example data set for the project.

This is the load step in Indexing.
"""

filename = 'data/companyPolicies.txt'
# url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

# # Use wget to download the file
# wget.download(url, out=filename)
# print('file downloaded')

# with open(filename, 'r') as file:
#     # Read the contents of the file
#     contents = file.read()
#     # print(contents)

"""
Splitting
LangChain is used in the indexing process to split documents
into smaller, manageable chunks. The default splitting method
(CharacterTextSplitter) uses \n\n as a separator, but this can
be customized. However, the splitting process is somewhat random,
which may cause inconsistencies. You can change it by adding the separator
parameter in the CharacterTextSplitter function; for example, separator="\n".
"""
loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(len(texts))

"""
Embedding
is the process of converting these text chunks into numerical values (vectors)
so the computer can recognize and search them efficiently.
"""
embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)  # store the embedding in docsearch using Chromadb
print('document ingested', docsearch)

# ============== Retrival Task ==================
"""
LLM model construction
"""

model_id = 'google/flan-ul2'
parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,  
    GenParams.MIN_NEW_TOKENS: 130, # this controls the minimum number of tokens in the generated output
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5 # this randomness or creativity of the model's responses
}

# Specify the path to your .env file
dotenv_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
load_dotenv(dotenv_path)

credentials = {
    "url": os.getenv("IBM_URL"),
    "apikey": os.getenv("IBM_API_KEY")
}
project_id = os.getenv("IBM_PROJECT_ID")

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

flan_ul2_llm = WatsonxLLM(model=model)

"""
Retrival
Integrating LangChain
LangChain has a number of components that are designed to help retrieve information
from the document and build question-answering applications,
which helps you complete the retrieve part of the Retrieval task.
"""
qa = RetrievalQA.from_chain_type(llm=flan_ul2_llm, 
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(), 
                                 return_source_documents=False)
query = "what is mobile policy?"
print(qa.invoke(query))
