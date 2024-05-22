import argparse
import os
import cv2
from flask import Flask, render_template, request, send_from_directory, url_for, Response, redirect
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DETECT_FOLDER = 'runs/detect'
OUTPUT_VIDEO = 'output.mp4'
PDF_FOLDER = 'pdfs'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DETECT_FOLDER):
    os.makedirs(DETECT_FOLDER)
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        print("No file part")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        extension = filename.rsplit('.', 1)[1].lower()

        if extension in ['jpg', 'jpeg', 'png']:
            return process_image(filepath, filename)
        elif extension == 'mp4':
            return process_video(filepath)
    return redirect(request.url)

def process_image(filepath, filename):
    try:
        model = YOLO('best.pt')
        img = cv2.imread(filepath)
        results = model(img, conf=0.1)
        
        output_path = os.path.join(DETECT_FOLDER, filename)
        detected_classes = []

        for result in results:
            res_plotted = result.plot()
            cv2.imwrite(output_path, res_plotted)

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    bbox = box.xyxy[0].tolist()
                    detected_classes.append({
                        'class': class_name,
                        'bbox': bbox
                    })

        generate_pdf(filename, detected_classes, filepath, output_path)
        return render_template('index.html', image_path=filename)
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error processing image", 500

def process_video(filepath):
    try:
        cap = cv2.VideoCapture(filepath)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30.0, (1280, 720))

        model = YOLO('best.pt')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            
            for result in results:
                res_plotted = result.plot()
                out.write(res_plotted)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        out.release()
        return redirect(url_for('video_feed'))
    except Exception as e:
        print(f"Error processing video: {e}")
        return "Error processing video", 500

def generate_pdf(filename, detected_classes, input_image_path, output_image_path):
    pdf_path = os.path.join(PDF_FOLDER, f"{filename}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    margin = 50

    c.drawString(margin, height - margin, "YOLOv8 Object Detection Results")
    c.drawString(margin, height - margin * 2, f"Image: {filename}")

    y_position = height - margin * 3

    # Add input image
    c.drawString(margin, y_position, "Input Image:")
    y_position -= 20

    input_image_reader = ImageReader(input_image_path)
    input_image_width, input_image_height = input_image_reader.getSize()
    input_aspect_ratio = input_image_width / input_image_height

    if input_aspect_ratio > 1:
        input_display_width = width - margin * 2
        input_display_height = input_display_width / input_aspect_ratio
    else:
        input_display_height = (height / 2) - margin
        input_display_width = input_display_height * input_aspect_ratio

    if y_position - input_display_height < margin:
        c.showPage()
        y_position = height - margin

    c.drawImage(input_image_reader, margin, y_position - input_display_height, width=input_display_width, height=input_display_height)
    y_position -= (input_display_height + 20)

    # Add detected classes and bounding boxes
    c.drawString(margin, y_position, "Detected Classes and Bounding Boxes:")
    y_position -= 20
    for detected in detected_classes:
        if y_position < margin:
            c.showPage()
            y_position = height - margin
        class_name = detected['class']
        bbox = detected['bbox']
        bbox_str = f"Bounding Box: {bbox}"
        c.drawString(margin, y_position, f"Class: {class_name}, {bbox_str}")
        y_position -= 20

    # Add output image
    if y_position - input_display_height < margin:
        c.showPage()
        y_position = height - margin

    c.drawString(margin, y_position, "Output Image:")
    y_position -= 20

    output_image_reader = ImageReader(output_image_path)
    output_image_width, output_image_height = output_image_reader.getSize()
    output_aspect_ratio = output_image_width / output_image_height

    if output_aspect_ratio > 1:
        output_display_width = width - margin * 2
        output_display_height = output_display_width / output_aspect_ratio
    else:
        output_display_height = (height / 2) - margin
        output_display_width = output_display_height * output_aspect_ratio

    if y_position - output_display_height < margin:
        c.showPage()
        y_position = height - margin

    c.drawImage(output_image_reader, margin, y_position - output_display_height, width=output_display_width, height=output_display_height)
    c.showPage()
    c.save()

@app.route('/download_pdf')
def download_pdf():
    try:
        pdf_files = sorted(os.listdir(PDF_FOLDER), key=lambda x: os.path.getctime(os.path.join(PDF_FOLDER, x)), reverse=True)
        latest_pdf = pdf_files[0]
        return send_from_directory(PDF_FOLDER, latest_pdf)
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return "Error downloading PDF", 500

@app.route('/display/<filename>')
def display_image(filename):
    try:
        return send_from_directory(DETECT_FOLDER, filename)
    except Exception as e:
        print(f"Error displaying image: {e}")
        return "Error displaying image", 500

def generate_video_frames():
    video = cv2.VideoCapture(OUTPUT_VIDEO)
    while True:
        success, frame = video.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
