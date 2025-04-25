import os
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from dotenv import load_dotenv
from functools import lru_cache
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load environment variables
def reload_env():
    load_dotenv("config.env", override=True)
    os.environ["N_MESSAGES"] = os.getenv("N_MESSAGES", "4")
    os.environ["TOP_K"] = os.getenv("TOP_K", "5")
    os.environ["GEMINI_MODEL"] = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")

def get_config_value(key, default=None, cast_func=str):
    return cast_func(os.getenv(key, default))

reload_env()
n_messages = get_config_value("N_MESSAGES", 4, int)
top_k = get_config_value("TOP_K", 5, int)
model_name = get_config_value("GEMINI_MODEL", "gemini-1.5-flash-latest", str)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

chat_history = []

def get_chat_history():
    return chat_history

# PDF Loading and Caching
def pdf_to_text(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@lru_cache(maxsize=1)
def load_all_policies_text(folder="."):
    full_text = ""
    found = False
    for file in os.listdir(folder):
        if file.lower().endswith(".pdf"):
            found = True
            full_path = os.path.join(folder, file)
            full_text += pdf_to_text(full_path) + "\n\n"
    if not found:
        logging.warning("No PDF files found in folder: " + folder)
    else:
        logging.info("Loaded all policy documents.")
    return full_text


def context_extractor(query, embedder, top_k=top_k):
    reload_env()
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    pdf_text = load_all_policies_text()

    if not pdf_text.strip():
        return "No policy documents found or loaded. Please check the folder and try again."

    paragraphs = pdf_text.split('\n\n')
    pdf_sentences = [p.replace('\n', ' ').strip() for p in paragraphs if p.strip()]

    if not pdf_sentences:
        return "No valid sentences found in the policy documents."

    bm25 = BM25Okapi(pdf_sentences)
    bm25_scores = bm25.get_scores(query)

    pdf_embeddings = embedder.encode(pdf_sentences, convert_to_tensor=True)
    transformer_hits = util.semantic_search(query_embedding, pdf_embeddings, top_k=top_k)[0]
    transformer_hits = [hit for hit in transformer_hits if hit['score'] > 0.1]

    combined_scores = {}
    for rank, hit in enumerate(transformer_hits, start=1):
        combined_scores[hit['corpus_id']] = 1 / rank

    for rank, idx in enumerate(sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k], start=1):
        if idx in combined_scores:
            combined_scores[idx] += 1 / rank
        else:
            combined_scores[idx] = 1 / rank

    top_sentences = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
    relevant_text = " ".join([pdf_sentences[idx] for idx in top_sentences])
    return relevant_text

def generate_text(messages, model=model_name):
    prompt = "\n".join([msg['content'] for msg in messages if msg['role'] == 'user'])
    model = genai.GenerativeModel(model)
    try:
        response = model.generate_content(prompt)
        logging.info("LLM response generated.")
        return response.text
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return "Sorry, there was an error processing your query."

def is_uncertain(response):
    uncertain_keywords = [
        "not sure", "I don’t know", "I'm not certain",
        "please contact", "reach out to", "I cannot", "unclear",
        "I recommend speaking to", "cannot find", "uncertain"
    ]
    return any(k in response.lower() for k in uncertain_keywords) or len(response.strip()) < 20

def generate_llm_response(query):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    history_str = ""
    if chat_history:
        for i, j in enumerate(chat_history[-n_messages:]):
            history_str += f"user message {i}: {j['user']} \n\n"
            history_str += f"ai response message {i}: {j['assistant']} \n\n"

    context = context_extractor(query, embedder)

    full_prompt = (
        f"you are a insurance policy question answering chatbot, you will have knowledge base in form of context now answer the user query about our company policies, treat the user as customer"
        f"previous chat history: {history_str} \n"
        f"The context is: {context}. \n\n"
        f"Now answer the question: {query}.\n\n"
        "Give a precise and user-friendly reply. if you are not certain tell them that you are uncertain about answeringS"
    )
    messages = [{"role": "user", "content": full_prompt}]
    response = generate_text(messages)

    if is_uncertain(response):
        fallback_msg = "I'm not fully confident in answering this. Let me connect you with a human support agent.\n here are the contact details \n Email: support@abcinsurance.com  \n Toll-Free: 1800-123-4567 \n Website: www.abcinsurance.com "
        logging.info("Fallback to human triggered.")
        return fallback_msg

    return response

def generate_llm_response_with_history(query):
    response = generate_llm_response(query)
    chat_history.append({
        "user": query,
        "assistant": response
    })
    return response

def save_his(filename="history.txt"):
    data = ""
    for i in chat_history:
        data += f"user_message: {i['user']} \n ai_response: {i['assistant']} \n\n"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data)
    logging.info("Chat history saved.")
    return 0

# For modular testing
def main():
    while True:
        user_input = input("Ask your insurance question (or type 'exit'): ")
        if user_input.lower() == "exit":
            save_his()
            break
        answer = generate_llm_response_with_history(user_input)
        print("\nAI:", answer)

# Optional: Uncomment to run as CLI
# if __name__ == "__main__":
#     main()
