import json, logging, time
from pathlib import Path
from queue import Queue
import threading
from typing import Tuple, Any
import fitz  

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import PictureItem

logging.basicConfig(level=logging.INFO)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_DIR = OUTPUT_DIR / "extracted_images"
IMG_DIR.mkdir(exist_ok=True)

def render_all_pages(pdf_path: Path, out_dir: Path) -> list[str]:
    """Render every page to PNG using PyMuPDF (guaranteed)."""
    paths = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=150)  
        out = out_dir / f"{pdf_path.stem}_page{i}.png"
        pix.save(out)
        paths.append(str(out))
    return paths

def convert_with_docling(pdf_path: str) -> Tuple[str, list[str], list[str], dict[str, Any], str]:
    pdf_path = Path(pdf_path)
    opts = PdfPipelineOptions(generate_images=True, generate_tables=True)
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    result = converter.convert(pdf_path)

    text_content = result.document.export_to_text()
    html_content = result.document.export_to_html()

    image_paths = render_all_pages(pdf_path, IMG_DIR)

    pic_counter = 1
    for item, _ in result.document.iterate_items():
        if isinstance(item, PictureItem):
            try:
                img = item.get_image(result.document)
                if img:
                    out = IMG_DIR / f"{pdf_path.stem}_pic_{pic_counter}.png"
                    img.save(out)
                    image_paths.append(str(out))
                    pic_counter += 1
            except Exception as e:
                logging.warning(f"Skipping embedded picture {pic_counter}: {e}")

    table_paths = []
    for idx, tbl in enumerate(result.document.tables, start=1):
        try:
            out = OUTPUT_DIR / f"{pdf_path.stem}_table{idx}.md"
            out.write_text(tbl.export_to_markdown(doc=result.document), encoding="utf-8")
            table_paths.append(str(out))
        except Exception as e:
            logging.warning(f"Skipping table {idx}: {e}")

    json_path = OUTPUT_DIR / f"{pdf_path.stem}_doc.json"
    json_path.write_text(json.dumps(result.document.export_to_dict(), indent=2), encoding="utf-8")

    return text_content, image_paths, table_paths, result.document.export_to_dict(), html_content

def worker(q: Queue):
    while not q.empty():
        pdf = q.get()
        try:
            logging.info(f"Processing {pdf} …")
            t, imgs, tbls, doc, html = convert_with_docling(pdf)
            stem = Path(pdf).stem
            (OUTPUT_DIR / f"{stem}.txt").write_text(t, encoding="utf-8")
            (OUTPUT_DIR / f"{stem}.yaml").write_text(json.dumps(doc, indent=2), encoding="utf-8")
            (OUTPUT_DIR / f"{stem}.html").write_text(html, encoding="utf-8")
            logging.info(f"Finished {pdf}, extracted {len(imgs)} images and {len(tbls)} tables")
        except Exception as e:
            logging.error(f"Error processing {pdf}: {e}")
        finally:
            q.task_done()

if __name__ == "__main__":
    start = time.time()
    pdfs = list(Path("data").glob("*.pdf"))
    if not pdfs:
        logging.warning("No PDF files found in ./data.")
    else:
        q = Queue()
        for p in pdfs:
            q.put(str(p))
        threads = [threading.Thread(target=worker, args=(q,)) for _ in range(min(4, len(pdfs)))]
        for t in threads: t.start()
        for t in threads: t.join()
    logging.info(f"Total time: {time.time()-start:.2f}s")
