# pdf_generator.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import os

PDF_FOLDER = 'pdfs'
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)

def generate_pdf(filename, detected_classes, image_path):
    pdf_path = os.path.join(PDF_FOLDER, f"{filename}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Add a title
    c.drawString(100, height - 50, "YOLOv8 Object Detection Results")
    c.drawString(100, height - 100, f"Image: {filename}")

    # Add detected classes
    y_position = height - 150
    c.drawString(100, y_position, f"Detected classes: {', '.join(detected_classes)}")

    # Add the image
    y_position -= 50  # Adjust position for image
    img_reader = ImageReader(image_path)
    img_width, img_height = img_reader.getSize()
    aspect_ratio = img_width / img_height

    max_img_width = width - 200
    max_img_height = height - 300

    if img_width > max_img_width:
        img_width = max_img_width
        img_height = img_width / aspect_ratio

    if img_height > max_img_height:
        img_height = max_img_height
        img_width = img_height * aspect_ratio

    c.drawImage(img_reader, 100, y_position - img_height, width=img_width, height=img_height)

    c.showPage()
    c.save()
