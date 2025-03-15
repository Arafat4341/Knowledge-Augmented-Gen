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

