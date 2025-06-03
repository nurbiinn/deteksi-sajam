from flask import Flask, render_template, request, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load semua model YOLO sekaligus
models = {
    'yolov8': YOLO('model/yolov8.pt'),
    'yolov9': YOLO('model/yolov9.pt'),
    'yolov10': YOLO('model/yolov10.pt')
}

# Label yang diizinkan (sesuaikan dengan kelas model)
allowed_labels = ['pisau', 'pistol']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_with_yolo(model, img_path_or_pil, save_path):
    """
    Jalankan prediksi YOLO pada gambar, simpan hasil deteksi dengan bounding box.
    Return: list hasil deteksi {label, confidence}
    """
    results = model(img_path_or_pil)
    boxes = results[0].boxes
    names = results[0].names

    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id].lower()
        conf = float(box.conf[0])
        if label in allowed_labels:
            detections.append({
                'label': label,
                'confidence': round(conf * 100, 2)
            })

    # Simpan gambar hasil deteksi dengan bounding box
    results[0].save(save_path)

    return detections

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/informasi')
def informasi():
    return render_template('informasi.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'image' not in request.files or 'model' not in request.form:
            return render_template('upload.html', error='File atau model belum dipilih')

        file = request.files['image']
        model_name = request.form['model']

        if model_name not in models:
            return render_template('upload.html', error='Model tidak valid')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Path untuk gambar hasil deteksi (disimpan dengan prefix 'result_')
            result_filename = 'result_' + filename
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

            # Jalankan deteksi dan simpan hasil bounding box
            detections = predict_with_yolo(models[model_name], filepath, result_path)

            return render_template('hasil_deteksi.html',
                                   filename=result_filename,
                                   detections=detections)
        else:
            return render_template('upload.html', error='Format file tidak didukung')

    return render_template('upload.html')

@app.route('/camera', methods=['GET', 'POST'])
def camera():
    if request.method == 'POST':
        data = request.get_json()
        if not data or 'image' not in data or 'model' not in data:
            return jsonify({'error': 'Data gambar atau model tidak ada'}), 400

        model_name = data['model']
        if model_name not in models:
            return jsonify({'error': 'Model tidak valid'}), 400

        image_data = data['image'].split(',')[1]
        img_data = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_data)).convert("RGB")

        debug_filename = 'camera_capture.png'
        debug_path = os.path.join(app.config['UPLOAD_FOLDER'], debug_filename)
        img.save(debug_path)

        result_filename = 'result_camera_capture.png'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

        detections = predict_with_yolo(models[model_name], img, result_path)

        return jsonify({
            'predicted': detections,
            'debug_image_url': url_for('static', filename='uploads/' + result_filename)
        })

    return render_template('camera.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
