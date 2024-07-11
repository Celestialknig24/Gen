import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        document = fitz.open(pdf_path)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text("text")
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return None

# Function to convert a PDF page to an image and extract text from the image
def extract_text_from_image(pdf_path, page_num):
    document = fitz.open(pdf_path)
    page = document.load_page(page_num)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img)
    return text

# Main function to process all PDFs in a folder
def process_pdfs_in_folder(folder_path):
    pdf_texts = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            if not text:  # If text extraction fails, try image-based extraction
                text = ""
                document = fitz.open(pdf_path)
                for page_num in range(len(document)):
                    text += extract_text_from_image(pdf_path, page_num)
            pdf_texts[filename] = text
    return pdf_texts

# Example usage
folder_path = "/path/to/your/pdf/folder"
pdf_texts = process_pdfs_in_folder(folder_path)

# Displaying extracted text for verification
for filename, text in pdf_texts.items():
    print(f"Text from {filename}:")
    print(text)
    print("\n\n")

# Further processing, e.g., converting to document objects for embeddings
# Assuming `Document` is a class that you use for embeddings
class Document:
    def __init__(self, text):
        self.text = text

documents = [Document(text) for text in pdf_texts.values()]

# Now you can proceed with embeddings on the `documents` list
