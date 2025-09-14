from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pytesseract
import subprocess
import os
import json
import torch


class VLMProcessor:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device='cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def caption_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption

    def ocr_image(self, img_path):
        txt = pytesseract.image_to_string(img_path)
        return txt.strip()

    def expand_caption(self, short_caption, ocr_text=None, ask_context=None):
        prompt = (
            "You are an assistant that writes detailed, vivid, factual image descriptions.\n\n"
            f"Short caption: {short_caption}\n"
        )
        if ocr_text:
            prompt += f"Detected text inside image: {ocr_text}\n"
        if ask_context:
            prompt += f"Extra context: {ask_context}\n"
        prompt += (
            "Write a complete descriptive paragraph (3-6 sentences) that covers: what is in the image,\n"
            "notable attributes (style, colors, objects), any readable text, and a short inference about the likely\n"
            "purpose of the image. Be factual, avoid hallucination, but use reasonable general knowledge.\n\nDescription:\n"
        )

        result = subprocess.run(["ollama", "run", "llama3"], input=prompt.encode('utf-8'), capture_output=True)
        return result.stdout.decode("utf-8").strip()

    def process_folder(self, folder_path):
        results = {}

        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            print(f"No images found. Created empty folder at {folder_path}")
            return results

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            print(f"No image files found in {folder_path}")
            return results

        for fname in sorted(files):
            path = os.path.join(folder_path, fname)
            try:
                short = self.caption_image(path)
            except Exception as e:
                short = f"(caption error: {e})"
            ocr = self.ocr_image(path)
            expanded = self.expand_caption(short, ocr_text=ocr)
            results[fname] = {"short": short, "ocr": ocr, "long": expanded}
            print(f"{fname} -> short: {short[:80]}... long: {expanded[:80]}...")
        return results


if __name__ == '__main__':
    images = "outputs/extracted_images"
    vlm = VLMProcessor(device='cpu')
    out = vlm.process_folder(images)

    if out:
        with open('outputs/image_captions.json', 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print('Captions saved')
    else:
        print("No captions saved because no images were found.")
