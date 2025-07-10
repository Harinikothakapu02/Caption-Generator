from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from flask import jsonify, Flask, request, render_template
import os
import time
from PIL import Image

app = Flask(__name__)

# Load the image-to-text model
def get_img_to_text_model():
    try:
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, feature_extractor, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

img_to_txt_model, feature_extractor, tokenizer, device = get_img_to_text_model()

class IMG_TO_TXT_CFG:
    max_length = 16
    num_beams = 4

def predict_step(image_paths):
    try:
        gen_kwargs = {"max_length": IMG_TO_TXT_CFG.max_length, "num_beams": IMG_TO_TXT_CFG.num_beams}
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = img_to_txt_model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

def generate_unique_filename(extension=".jpg"):
    timestamp = int(time.time())
    return f"image_{timestamp}{extension}"

def save_image(image_file):
    try:
        upload_dir = "static/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        unique_filename = generate_unique_filename(extension=os.path.splitext(image_file.filename)[1])
        image_path = os.path.join(upload_dir, unique_filename)
        image_file.save(image_path)
        return image_path
    except Exception as e:
        print(f"Error saving image: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/caption', methods=['POST'])
def caption():
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image_path = save_image(image_file)

    try:
        generated_text = predict_step([image_path])
        response = {"generated_text": generated_text[0]}
        return jsonify(response), 200
    except Exception as e:
        print(f"Error in /caption route: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)