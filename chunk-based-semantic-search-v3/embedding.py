import transformers
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
model.eval()


def get_embedding(text):
    inputs = tokenizer(text, return_tensors= 'pt', padding = True, trunction = True)

    with torch.no_grad():
        outputs = model(**inputs)

    mask = inputs['attention_mask'].unsqueeze(-1)
    
    masked_embeddings = outputs.last_hidden_state * mask
    
    summed_embeddings = masked_embeddings.sum(dim = 1)

    mask_sum = mask.sum(dim = 1).clamp(min = 1e-9)
    embedding = summed_embeddings / mask_sum

    return embedding
