from flask import Flask, render_template, request, url_for, redirect
from utils import TumorSegmentationModel, TumorReportGenerator
import os
from PIL import Image
import uuid
import time
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load model once
MODEL_PATH = './models/best_model.weights.h5'
segmenter = TumorSegmentationModel(model_path=MODEL_PATH)
report_gen = TumorReportGenerator()

# Temporary in-memory cache (not persistent)
cache = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_time = time.time()
        if 'image' not in request.files:
            return render_template('index.html', error="No file uploaded")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        try:
            # Generate unique ID
            image_id = str(uuid.uuid4())[:8]
            base_filename = f"{os.path.splitext(file.filename)[0]}_{image_id}"
            image_filename = f"{base_filename}.png"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

            # Save original image
            img = Image.open(file).convert('RGB')
            img.save(image_path)

            # Segment
            input_tensor, img_for_display = segmenter.preprocess_image(image_path)
            binary_mask = segmenter.predict_mask(input_tensor)
            overlayed = segmenter.overlay_mask_on_image(img_for_display, binary_mask)

            # Save overlayed image
            result_filename = f"result_{base_filename}.png"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            Image.fromarray(overlayed).save(result_path)

            original_img_url = url_for('static', filename=f'uploads/{image_filename}')
            result_img_url = url_for('static', filename=f'results/{result_filename}')

            # Store data in memory
            cache[image_id] = {
                "mask": binary_mask,
                "image_shape": img_for_display.shape,
                "original_img": original_img_url,
                "overlayed_img": result_img_url,
                "width": img.width,
                "height": img.height,
                "start_time": start_time
            }

            # Stats without clinical report
            tumor_pixels = np.sum(binary_mask)
            total_pixels = img.width * img.height
            area_percent = (tumor_pixels / total_pixels) * 100

            stats = {
                'tumor_area': round(area_percent, 2),
                'confidence': round(float(binary_mask.max()) * 100, 2),
                'resolution': f"{img.width}x{img.height}",
                'processing_time': round(time.time() - start_time, 2)
            }

            return render_template('index.html',
                                   original_img=original_img_url,
                                   overlayed_img=result_img_url,
                                   stats=stats,
                                   image_id=image_id)

        except Exception as e:
            return render_template('index.html', error=f"Error processing image: {str(e)}")

    return render_template('index.html')


@app.route('/generate_report', methods=['POST'])
def generate_report():
    image_id = request.form.get("image_id")
    if not image_id or image_id not in cache:
        return render_template('index.html', error="Session expired or invalid image ID.")

    try:
        data = cache[image_id]
        mask = data["mask"]
        image_shape = data["image_shape"]

        features = report_gen.extract_tumor_features(mask, image_shape)

        prompt = report_gen.generate_llm_report(features) + \
                 "\n\nPlease provide only a clinical summary without follow-up questions or interaction prompts."
        clinical_report = report_gen.call_llm_api(prompt)
        if any(word in clinical_report.lower() for word in ["charlie", "alice", "bob", "logic puzzle", "cannot determine","i'm sorry, but as an ai"]):
            clinical_report = "Error: Invalid LLM output. Please try again or review the model behavior."


        tumor_pixels = np.sum(mask)
        total_pixels = data['width'] * data['height']
        area_percent = (tumor_pixels / total_pixels) * 100

        stats = {
            'tumor_area': round(area_percent, 2),
            'confidence': round(float(mask.max()) * 100, 2),
            'resolution': f"{data['width']}x{data['height']}",
            'processing_time': round(time.time() - data['start_time'], 2)
        }

        return render_template('index.html',
                               original_img=data["original_img"],
                               overlayed_img=data["overlayed_img"],
                               stats=stats,
                               clinical_report=clinical_report,
                               image_id=image_id)

    except Exception as e:
        return render_template('index.html', error=f"Error generating report: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
