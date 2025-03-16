from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def build_conversational_chain(llm, retriever):
    memory = ConversationBufferMemory(memory_key="chat_history", return_message=True)

    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                               chain_type="stuff",
                                               retriever=retriever,
                                               memory=memory,
                                               get_chat_history=lambda h: h,
                                               return_source_documents=False)
    return qa
