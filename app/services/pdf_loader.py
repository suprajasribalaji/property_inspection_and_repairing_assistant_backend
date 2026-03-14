from pypdf import PdfReader
import re
import os

def load_predefined_questions():

    # Construct an absolute path relative to this script's directory
    # This script is in backend/app/services, so we go up one directory to app, then into data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, "data", "questions.pdf")
    
    reader = PdfReader(pdf_path)

    questions = []

    for page in reader.pages:
        text = page.extract_text()

        lines = text.split("\n")

        for line in lines:

            line = line.strip()

            # match numbered questions like "1 Are there cracks..."
            match = re.match(r"^\d+\s+(.*)", line)

            if match:
                question = match.group(1).strip()
                questions.append(question)

    return questions 