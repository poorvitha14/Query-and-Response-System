import os
import json
from sentence_transformers import SentenceTransformer
import faiss

EMBED_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 400
OVERLAP = 80

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        pieces = tokens[i:i+size]
        chunks.append(' '.join(pieces))
        i += size - overlap
    return chunks

def build_index(text_file, captions_json, tables_json_dir='outputs/tables_json', out='outputs/faiss_index.pkl'):
    embedder = SentenceTransformer(EMBED_MODEL)
    sentences = []
    metadata = []

    with open(text_file,'r',encoding='utf-8') as f:
        raw = f.read()
    chunks = chunk_text(raw)
    for i,ch in enumerate(chunks):
        sentences.append(ch)
        metadata.append({'type':'text','source':'pdf','chunk':i})

    with open(captions_json,'r',encoding='utf-8') as f:
        caps = json.load(f)
    for img,info in caps.items():
        sentences.append(info.get('long') or info.get('short'))
        metadata.append({'type':'image','source':img})

    if os.path.isdir(tables_json_dir):
        for fname in sorted(os.listdir(tables_json_dir)):
            if fname.endswith('.json'):
                path = os.path.join(tables_json_dir,fname)
                with open(path,'r',encoding='utf-8') as f:
                    table_rows = json.load(f)
                
                for r in table_rows:
                    row_text = ' | '.join([str(v) for v in r.values()])
                    sentences.append(row_text)
                    metadata.append({'type':'table','source':fname})

    import numpy as np
    emb = embedder.encode(sentences, show_progress_bar=True)
    d = emb.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(emb).astype('float32'))

    import pickle
    with open(out,'wb') as f:
        pickle.dump({'index':index,'sentences':sentences,'metadata':metadata},f)
    print('FAISS index built')

if __name__=='__main__':
    build_index('outputs/extracted_text.txt','outputs/image_captions.json')
