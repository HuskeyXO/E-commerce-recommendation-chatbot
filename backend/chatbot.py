import pandas as pd
import os
import time
import numpy as np
import faiss
from dotenv import load_dotenv
import cohere
from langchain.chains import RetrievalQA, LLMChain
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.llms import Cohere

# Load environment variables
load_dotenv("C:\\Users\\rasto\\OneDrive\\Study Room\\AFTER JEE\\UNDERGRAD@BMU\\Semester 5\\Artificial Intelligence\\ecommerce-product-recommendation\\backend\\.env")

# Access the Cohere API key
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY is not set. Ensure it is defined in the .env file.")

# Initialize Cohere client
co = cohere.Client(cohere_api_key)

# Load dataset
df = pd.read_csv('data/bq-results-20240205-004748-1707094090486.csv').head(100)

# Preprocess data for embedding
df['combined_info'] = df.apply(
    lambda row: f"Product: {row['product_name']}. Product Department: {row['product_department']}. "
                f"Price: ${row['sale_price']}. Stock quantity: {row['stock_quantity']}",
    axis=1
)

# Load processed dataset
loader = DataFrameLoader(df, page_content_column="combined_info")
docs = loader.load()

# Document splitting
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = [doc.page_content for doc in text_splitter.split_documents(docs)]

# Define the Cohere embedding function
def cohere_embedding_function(text):
    response = co.embed(
        texts=[text],
        model="embed-english-v2.0",
        truncate="RIGHT"
    )
    return response.embeddings[0]

# Batch embedding generation
def generate_embeddings_with_cohere(texts, batch_size=50):
    embeddings = []
    print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = co.embed(
                texts=batch,
                model="embed-english-v2.0",
                truncate="RIGHT"
            )
            embeddings.extend(response.embeddings)
        except cohere.errors.TooManyRequestsError:
            print(f"Rate limit exceeded. Waiting for 60 seconds...")
            time.sleep(60)
            response = co.embed(
                texts=batch,
                model="embed-english-v2.0",
                truncate="RIGHT"
            )
            embeddings.extend(response.embeddings)
    return embeddings

# Generate embeddings
embeddings = generate_embeddings_with_cohere(texts)

# Ensure embeddings and texts are properly aligned
assert len(texts) == len(embeddings), "The number of texts and embeddings must match."

# Create FAISS index, docstore, and mappings
dimension = len(embeddings[0])
faiss_index = faiss.IndexFlatL2(dimension)
docstore = InMemoryDocstore()
index_to_docstore_id = {}

# Assign IDs dynamically, add documents and embeddings
for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    doc_id = str(i)  # Generate document ID as string
    docstore._dict[doc_id] = {"text": text}  # Add document to docstore
    faiss_index.add(np.array([embedding]).astype('float32'))  # Add embedding to FAISS
    index_to_docstore_id[faiss_index.ntotal - 1] = doc_id  # Map FAISS index ID to docstore ID

# Verify synchronization
assert len(docstore._dict) == len(index_to_docstore_id), "Mismatch between docstore and index mappings."
assert faiss_index.ntotal == len(docstore._dict), "FAISS index size does not match docstore size."

print(f"Docstore count: {len(docstore._dict)}")
print(f"FAISS index size: {faiss_index.ntotal}")

# Initialize FAISS vector store
vectorstore = FAISS(
    embedding_function=cohere_embedding_function,
    docstore=docstore,
    index=faiss_index,
    index_to_docstore_id=index_to_docstore_id,
)

# Safe retrieval function
def safe_retrieve(query):
    try:
        # Generate query embedding
        query_embedding = cohere_embedding_function(query)
        query_embedding = np.array([query_embedding]).astype("float32")

        # Perform FAISS search
        distances, indices = faiss_index.search(query_embedding, k=3)
        print(f"Query: {query}")
        print(f"Nearest Neighbor Distances: {distances}")
        print(f"Nearest Neighbor Indices: {indices}")

        results = []
        for i, idx in enumerate(indices[0]):  # Loop over FAISS result indices
            if idx >= 0:  # Check if the index is valid
                doc_id = index_to_docstore_id.get(idx)
                document = docstore._dict.get(doc_id)

                if document:
                    # Append text and distance (converted to score for clarity)
                    score = distances[0][i]
                    results.append((document['text'], score))
                else:
                    print(f"Failed to fetch document for FAISS ID {idx}, Docstore ID: {doc_id}")

        # Process and format results
        if results:
            return "\n".join([f"{text} (Score: {score:.2f})" for text, score in results])
        else:
            return "No relevant documents found."

    except Exception as e:
        print(f"Error during retrieval: {e}")
        return "An error occurred while processing your request."

# Prompt templates
manual_template = """ 
Kindly suggest three similar products based on the description I have provided below:
Product Department: {department},
Product Category: {category},
Product Brand: {brand},
Maximum Price range: {price}.
Please provide complete answers including product department name, product category, product name, price, and stock quantity.
"""
prompt_manual = PromptTemplate(
    input_variables=["department", "category", "brand", "price"],
    template=manual_template,
)

chatbot_template = """ 
You are a friendly, conversational retail shopping assistant that helps customers to find products that match their preferences.
From the following context and chat history, assist customers in finding what they are looking for based on their input. 
For each question, suggest three products, including their category, price, and current stock quantity.
Sort the answer by the cheapest product.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
chat history: {history}
input: {question}
Your Response:
"""
chatbot_prompt = PromptTemplate(
    input_variables=["context", "history", "question"],
    template=chatbot_template,
)

# Memory
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Initialize Cohere LLM
llm = Cohere(model="command-xlarge", temperature=0.7, client=co)

# Define QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": chatbot_prompt,
    }
)

# Define Manual input chain
chain = LLMChain(
    llm=llm,
    prompt=prompt_manual,
    verbose=True,
)

# Check document count
print(f"Total documents in docstore: {len(docstore._dict)}")
print(f"Total entries in FAISS index: {faiss_index.ntotal}")
print(f"Total mappings in index_to_docstore_id: {len(index_to_docstore_id)}")

# Check FAISS index stats
print(f"FAISS index size: {faiss_index.ntotal}")

# Check document mappings
print("Index to Docstore ID Mappings:")
for index_id, doc_id in index_to_docstore_id.items():
    doc = docstore._dict.get(doc_id)
    print(f"FAISS Index ID: {index_id}, Docstore ID: {doc_id}, Document: {doc}")

# Verify docstore contents
print("Docstore Contents:")
for doc_id, doc in docstore._dict.items():
    print(f"Docstore ID: {doc_id}, Document: {doc}")


faiss_id = 86
doc_id = index_to_docstore_id.get(faiss_id)  # Map FAISS index ID to docstore ID
document = docstore._dict.get(doc_id)        # Retrieve document from docstore

print(f"FAISS ID: {faiss_id}")
print(f"Mapped Docstore ID: {doc_id}")
print(f"Document: {document}")

query = "Buffalo by David Bitton Men's Six Jeans"
response = safe_retrieve(query)
print("Response for query:", response)

print(f"index_to_docstore_id: {index_to_docstore_id}")
print(f"docstore._dict: {docstore._dict}")


print("Chatbot initialized successfully!")
