import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_PATH = "vectorstore"


def create_or_load_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # If vectorstore already exists → load it
    if os.path.exists(VECTOR_PATH):
        vectorstore = FAISS.load_local(
            VECTOR_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        # Create new vectorstore
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local(VECTOR_PATH)

    return vectorstore
