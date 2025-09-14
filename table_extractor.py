import os
import pandas as pd

def canonicalize_tables(tables_dir, out_dir='outputs/tables_json'):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(tables_dir):
        if fname.lower().endswith('.csv'):
            path = os.path.join(tables_dir, fname)
            try:
                df = pd.read_csv(path)
            except Exception:
                df = pd.read_csv(path, encoding='latin1')
            
            df = df.fillna('')
            out_path = os.path.join(out_dir, fname.replace('.csv','.json'))
            df.to_json(out_path, orient='records')
    print('Tables canonicalized')

if __name__=='__main__':
    canonicalize_tables('outputs/extracted_tables')
