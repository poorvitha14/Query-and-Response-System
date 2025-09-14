# Query-and-Response-System
An AI-powered Question Answering system that extracts and interprets information from semantic and structured multimodal documents, including text, tables, charts, and images. Built with FAISS-based retrieval, Visual-Language Models, a Gradio interface, and Ollama-hosted LLMs.

Project Workflow:
Step 1: PDF Ingestion
* Create a folder named data/ in the project root.
* Place the input PDF file (e.g., sample.pdf) into this folder.
* This folder acts as the entry point for all documents to be processed.

Step 2: Environment Setup
* Create a Python virtual environment (venv).
* Install dependencies using: pip install -r requirements.txt
* Ensure system dependencies are installed:
    * Tesseract OCR → Required for recognizing text inside images.
    * Poppler → Required by pdf2image for PDF rendering.

Step 3: Content Extraction (process_pdf.py)
* Runs the Docling DocumentConverter on the PDF.
* Produces multiple outputs inside the outputs/ folder:
    * Plain Text (.txt) – Extracted raw text.
    * HTML (.html) – Preserves document layout and formatting.
    * JSON / YAML (.json, .yaml) – Structural metadata of the PDF.
    * Tables (.md) – Each table is exported into Markdown.
    * Images (outputs/extracted_images/) – Extracted full-page and embedded images.

Step 4: Image Captioning (vlm_processor.py)
* Processes the extracted images.
* For each image:
    * Generates captions using a Vision-Language Model (BLIP).
    * Runs OCR with Tesseract to extract embedded text.
    * Expands short captions into detailed descriptions with an LLM.
* Results are stored in outputs/image_captions.json.

Step 5: Table Structuring (table_extractor.py)
* Extracts tabular data using Camelot.
* Converts raw tables into cleaned and canonicalized structures.
* Stores results in structured formats for downstream retrieval.

Step 6: Embedding & Indexing (retriever.py)
* Creates embeddings for:
    * Text content
    * Structured tables
    * Image captions
* Uses Sentence-Transformers for vector embeddings.
* Stores them inside a FAISS vector index for fast semantic search.

Step 7: Interactive Search (app.py)
* Launches a Gradio-based UI.
* Users can ask natural language questions such as:
    * “Summarize page 2.”
    * “What does Figure 3 describe?”
    * “Show me the values from the sales table.”
* The system retrieves the most relevant content (text, table, or image) from the FAISS index and generates a natural language answer.
