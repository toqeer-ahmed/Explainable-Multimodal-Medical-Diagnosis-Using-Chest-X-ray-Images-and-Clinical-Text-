import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class ImageGradCamWrapper(torch.nn.Module):
    """
    Wrapper for MultimodalFusion model to be used with Grad-CAM.
    Grad-CAM expects a model that takes only the image as input,
    so we fix the text inputs (input_ids, attention_mask).
    """
    def __init__(self, multimodal_model, input_ids, attention_mask):
        super().__init__()
        self.multimodal_model = multimodal_model
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def forward(self, images):
        return self.multimodal_model(images, self.input_ids, self.attention_mask)

def generate_gradcam(model, image_tensor, input_ids, attention_mask, original_image, target_class=None, device='cpu'):
    """
    Generates a Grad-CAM heatmap for the given image and text inputs.
    
    Args:
        model: The trained MultimodalFusion model.
        image_tensor: Normalized image tensor (1, 3, 224, 224).
        input_ids: Tokenized text tensor (1, max_len).
        attention_mask: Attention mask tensor (1, max_len).
        original_image: Unnormalized numpy array of the image (224, 224, 3) with values in [0, 1] for visualization.
        target_class: Integer index of the class to explain. If None, explains the highest predicted class.
        device: 'cpu' or 'cuda'.
    
    Returns:
        visualization: Numpy array of the heatmap overlaid on the original image.
        target_class: The class index that was explained.
    """
    model.eval()
    
    # Wrap model to only take image as input
    wrapper_model = ImageGradCamWrapper(model, input_ids.to(device), attention_mask.to(device)).to(device)
    wrapper_model.eval()
    
    # Target layer for DenseNet121
    # DenseNet features block ends with a batch norm, we target the last conv block
    target_layers = [model.image_encoder.features[-1]]
    
    # Determine target class if not provided
    if target_class is None:
        with torch.no_grad():
            logits = wrapper_model(image_tensor.to(device))
            target_class = torch.argmax(logits, dim=1).item()
            
    # Define targets
    targets = [ClassifierOutputTarget(target_class)]
    
    # Generate CAM
    with GradCAM(model=wrapper_model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor.to(device), targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay heatmap on original image
        visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
        
    return visualization, target_class

def get_text_attention(model, input_ids, attention_mask, tokenizer, device='cpu'):
    """
    Extracts text attention weights from the BERT encoder.
    
    Args:
        model: The trained MultimodalFusion model.
        input_ids: Tokenized text tensor (1, max_len).
        attention_mask: Attention mask tensor (1, max_len).
        tokenizer: The BERT tokenizer.
        device: 'cpu' or 'cuda'.
        
    Returns:
        tokens: List of strings.
        attention_scores: 1D numpy array of attention scores per token.
    """
    model.eval()
    with torch.no_grad():
        # Get BERT output with attentions
        outputs = model.text_encoder.bert(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_attentions=True
        )
        
    # outputs.attentions is a tuple of 12 layers
    # Each layer is (batch_size, num_heads, sequence_length, sequence_length)
    # We take the last layer
    last_layer_attention = outputs.attentions[-1]
    
    # Average across all attention heads
    # Shape: (batch_size, sequence_length, sequence_length)
    avg_attention = torch.mean(last_layer_attention, dim=1)
    
    # Get attention weights of the [CLS] token (index 0) with respect to all other tokens
    # Shape: (sequence_length,)
    cls_attention = avg_attention[0, 0, :].cpu().numpy()
    
    # Get tokens
    input_ids_list = input_ids[0].cpu().numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
    
    # Filter out special tokens and padding
    valid_indices = [i for i, token in enumerate(tokens) if token not in ['[PAD]', '[CLS]', '[SEP]']]
    
    valid_tokens = [tokens[i] for i in valid_indices]
    valid_scores = [cls_attention[i] for i in valid_indices]
    
    # Normalize scores for better visualization
    if len(valid_scores) > 0:
        valid_scores = np.array(valid_scores)
        valid_scores = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min() + 1e-9)
    
    return valid_tokens, valid_scores

def visualize_explainability(image_vis, tokens, attention_scores, class_name, save_path=None):
    """
    Plots the Grad-CAM image and a bar chart of top text attention scores side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Plot Grad-CAM Image
    axes[0].imshow(image_vis)
    axes[0].axis('off')
    axes[0].set_title(f'Grad-CAM Heatmap\nTarget Class: {class_name}', fontsize=14)
    
    # 2. Plot Text Attention
    # Get top 15 words for readability
    if len(tokens) > 15:
        top_indices = np.argsort(attention_scores)[-15:]
        top_tokens = [tokens[i] for i in top_indices]
        top_scores = [attention_scores[i] for i in top_indices]
    else:
        top_tokens = tokens
        top_scores = attention_scores
        
    axes[1].barh(top_tokens, top_scores, color='skyblue')
    axes[1].set_xlabel('Attention Score')
    axes[1].set_title('BERT Self-Attention ([CLS] Token)', fontsize=14)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved explainability visualization to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Explainability module loaded.")
