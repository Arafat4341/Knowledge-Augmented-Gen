from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

def load_and_split(filename, chunk_size=1000):
    loader = TextLoader(filename)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return splitter.split_documents(documents)

def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings()
    return Chroma.from_documents(docs, embeddings)
