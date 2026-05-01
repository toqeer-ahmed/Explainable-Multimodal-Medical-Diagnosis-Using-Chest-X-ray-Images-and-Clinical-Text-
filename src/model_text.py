import torch
import torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', freeze_bert=False):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        # BERT base outputs a 768-dimensional vector
        self.feature_dim = 768

    def forward(self, input_ids, attention_mask):
        # input_ids shape: (batch_size, max_len)
        # attention_mask shape: (batch_size, max_len)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # We extract the [CLS] token representation to use as the sentence/report embedding
        # outputs.pooler_output is the [CLS] representation (batch_size, 768)
        cls_output = outputs.pooler_output
        return cls_output

if __name__ == "__main__":
    model = TextEncoder()
    # Dummy tokens
    dummy_input_ids = torch.randint(0, 1000, (2, 128))
    dummy_mask = torch.ones(2, 128)
    output = model(dummy_input_ids, dummy_mask)
    print(f"Text Encoder output shape: {output.shape}")
