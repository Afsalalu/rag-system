# importing requirements
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

#input text document file
documents = [
    "Artificial Intelligence enables machines to mimic human intelligence such as learning and reasoning.",
    "Machine Learning is a subset of Artificial Intelligence that allows systems to learn from data without explicit programming.",
    "Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers.",
    "Natural Language Processing helps machines understand and generate human language.",
    "Retrieval-Augmented Generation combines document retrieval with text generation to improve accuracy."
]


#chunking data
def chunk_text(text, chunk_size=20, overlap=5):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


chunks = []
for doc in documents:
    chunks.extend(chunk_text(doc))


#embedding the model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedding_model.encode(chunks)

#retrieving the chunks
def retrieve_chunks(query, top_k=2):
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

#finding answer using transformer
def transformer_answer(question):
    retrieved_chunks = retrieve_chunks(question)
    return " ".join(retrieved_chunks)

#finding answer using LLM Model Groq
client = Groq(api_key=os.getenv("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"))

def llm_answer(question):
    retrieved_chunks = retrieve_chunks(question)
    context = "\n".join(retrieved_chunks)

    system_prompt = (
        "You are an AI assistant. Answer strictly using the given context. "
        "If the answer is not present, say 'I don't know'."
    )

    user_prompt = f"""
Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

#getting user input question
q = input("Enter the Question:\t")

#printing Transformer Output
print("\nTransformer-based Answer:")
print(transformer_answer(q))

#printing LLM Model Output
print("\nLLM-based Answer:")
print(llm_answer(q))
