import os
import io
import base64
import torch
import pandas as pd
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms
from transformers import BertTokenizer

from src.fusion_model import MultimodalFusion
from src.explainability import generate_gradcam, get_text_attention

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Try to load model
try:
    print("Loading PyTorch model and tokenizers...")
    df_labels = pd.read_csv('data/processed/labels.csv')
    classes = df_labels['class_name'].tolist()
    num_classes = len(classes)
    
    model = MultimodalFusion(num_classes=num_classes).to(device)
    try:
        model.load_state_dict(torch.load('outputs/models/best_model_V2.pth', map_location=device))
        print("✅ Trained model loaded successfully and ready for predictions!")
    except Exception as e:
        print(f"⚠️ Warning: Could not find 'outputs/models/best_model_V2.pth'. Using UNTRAINED weights to demonstrate the pipeline! ({e})")
        
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    
    # Standard ResNet/DenseNet normalization
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
except Exception as e:
    print(f"❌ Fatal Error initializing system: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Ensure best_model_V2.pth is in outputs/models/'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    text = request.form.get('text', '')

    try:
        # 1. Process image
        img = Image.open(file).convert('RGB')
        img_tensor = img_transform(img).unsqueeze(0).to(device) # Shape: (1, 3, 224, 224)
        
        # Prepare image for visualization (un-normalize)
        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        img_np = img_tensor[0].cpu().numpy()
        img_unnorm = (img_np * std) + mean
        img_unnorm = np.transpose(img_unnorm, (1, 2, 0))
        img_unnorm = np.clip(img_unnorm, 0, 1)

        # 2. Process text
        if text.strip():
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
        else:
            # Dummy text for image-only (all PAD tokens)
            input_ids = torch.zeros((1, 128), dtype=torch.long).to(device)
            attention_mask = torch.zeros((1, 128), dtype=torch.long).to(device)

        # 3. Model Inference
        with torch.no_grad():
            logits = model(img_tensor, input_ids, attention_mask)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        # Get top 5 predictions
        top_indices = np.argsort(probs)[-5:][::-1]
        predictions = [{'label': classes[i], 'confidence': float(probs[i])} for i in top_indices]

        # 4. Explainability: Grad-CAM
        target_idx = top_indices[0] # Explain the top prediction
        vis_image, _ = generate_gradcam(
            model=model, 
            image_tensor=img_tensor, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            original_image=img_unnorm,
            target_class=int(target_idx),
            device=device
        )

        # Convert Grad-CAM heatmap to base64 for React frontend
        vis_image_uint8 = (vis_image * 255).astype(np.uint8)
        vis_pil = Image.fromarray(vis_image_uint8)
        buffered = io.BytesIO()
        vis_pil.save(buffered, format="JPEG")
        heatmap_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        heatmap_url = f"data:image/jpeg;base64,{heatmap_b64}"

        # 5. Explainability: BERT Text Attention
        highlighted_text = []
        if text.strip():
            tokens, att_scores = get_text_attention(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                device=device
            )
            for w, s in zip(tokens, att_scores):
                w_clean = w.replace('##', '') # Clean subwords
                highlighted_text.append({'word': w_clean + " ", 'score': float(s)})

        return jsonify({
            'predictions': predictions,
            'heatmap_url': heatmap_url,
            'highlighted_text': highlighted_text
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
