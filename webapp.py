import argparse
import os
import cv2
import time
from flask import Flask, render_template, request, send_from_directory, url_for, Response, redirect
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DETECT_FOLDER = 'runs/detect'
OUTPUT_VIDEO = 'output.mp4'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DETECT_FOLDER):
    os.makedirs(DETECT_FOLDER)

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
        
        for result in results:
            res_plotted = result.plot()
            output_path = os.path.join(DETECT_FOLDER, filename)
            cv2.imwrite(output_path, res_plotted)

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
    parser = argparse.ArgumentParser(description="Flask app exposing YOLOv9 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)

