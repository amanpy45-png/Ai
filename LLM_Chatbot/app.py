import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

text = "Once upon a time"

tokens = tokenizer(text, return_tensors= 'pt')


model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
print('model loaded successfully')


# give tokens to the model
with torch.no_grad(): # we're only using model,no training, so pytorch doesn't need to store grad
    outputs = model(**tokens)

print(type(outputs))
print(outputs.keys())
print(outputs.logits.shape)


print(len(outputs.logits[0]))
print(outputs.logits[0][0].shape)
print(outputs.logits[0][1].shape)
print(outputs.logits[0][2].shape)

# last token
print(outputs.logits[0][-1])
print(outputs.logits[0][-1].shape)

# convert those 32,000 scores into one predicted token.
next_token = torch.argmax(outputs.logits[0][-1])# return the idx of higest score
# pred token id
print(next_token)


# convert that ID back to text.
print(tokenizer.decode(next_token))


print(tokens['input_ids'].shape)
print(next_token.shape)

# making it a 2d 
# print(next_token.unsqueeze(0).unsqueeze(0).shape)


# append token
new_input_ids = torch.cat(
    [tokens['input_ids'], next_token.unsqueeze(0).unsqueeze(0)],
    dim = 1
)
print(new_input_ids)
print(new_input_ids.shape)


print(tokenizer.decode(new_input_ids[0]))
# print(tokenizer.eos_token_id)

# generation
for _ in range(20):
    outputs = model(input_ids=new_input_ids)
    last_token_logits = outputs.logits[0][-1]

    # print(last_token_logits[:10])
    temp = 2.0
    probs = torch.softmax(last_token_logits / temp, dim=-1)
    # print(torch.max(probs))
    # print(torch.argmax(probs))
    
    next_token = torch.multinomial(probs, num_samples=1).squeeze()
    
    print(next_token.item())
    print(tokenizer.decode(next_token))

    if next_token == tokenizer.eos_token_id:
        break

    new_input_ids = torch.cat(
        [new_input_ids, next_token.unsqueeze(0).unsqueeze(0)],
        dim=1
    )





