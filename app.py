import gradio as gr
import subprocess
import pickle
from sentence_transformers import SentenceTransformer

EMBED_MODEL = 'all-MiniLM-L6-v2'
TOP_K = 6

embedder = SentenceTransformer(EMBED_MODEL)

with open('outputs/faiss_index.pkl','rb') as f:
    data = pickle.load(f)
index = data['index']
sentences = data['sentences']
metadata = data['metadata']

import numpy as np

def semantic_retrieve(question, k=TOP_K):
    q_emb = embedder.encode([question]).astype('float32')
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        results.append({'text': sentences[idx], 'meta': metadata[idx]})
    return results

PROMPT_TEMPLATE = '''You are a helpful assistant. Use the provided context from a PDF (and images/tables) to answer the user's question.
If the question asks for an exact table cell but the user doesn't know headers, try to find relevant rows by approximate values or synonyms.
If the context isn't sufficient, use your general knowledge but say when you are inferring.

Context:
{context}

Question: {question}
Answer:'''

def call_llm(prompt):
    res = subprocess.run(['ollama','run','llama3'], input=prompt.encode('utf-8'), capture_output=True)
    return res.stdout.decode('utf-8')

def answer(query):
    # 1) retrieve
    hits = semantic_retrieve(query)
    ctx = ''
    for h in hits:
        ctx += f"[source:{h['meta']} ] {h['text']}\n\n"

    # 2) create prompt
    prompt = PROMPT_TEMPLATE.format(context=ctx, question=query)
    out = call_llm(prompt)
    return out

# Gradio
iface = gr.Interface(fn=answer, inputs=gr.Textbox(lines=2,placeholder='Ask any question about the PDF...'), outputs='text', title='PDF QA (Dockling + rich captions)')
iface.launch()
