from embeddings.embedder import load_and_split, create_vectorstore
from llm.watsonx_llm import load_watsonx_llm
from retriever.retriever import create_retriever
from chain.qa_chain import build_conversational_chain

def main():
    filename = 'data/companyPolicies.txt'
    docs = load_and_split(filename)
    vectorstore = create_vectorstore(docs)
    retriever = create_retriever(vectorstore)
    llm = load_watsonx_llm()

    qa = build_conversational_chain(llm, retriever)

    print("Welcome! Ask your questions (type 'exit' to quit):")
    history = []
    while True:
        query = input("Question: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        result = qa({"question": query}, {"chat_history": history})
        history.append((query, result["answer"]))
        print("Answer:", result["answer"])

if __name__ == "__main__":
    main()
