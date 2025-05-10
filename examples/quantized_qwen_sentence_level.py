import torch
from transformers import AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2
from transformers import BitsAndBytesConfig

from lxt.efficient import monkey_patch
from lxt.utils import pdf_heatmap, clean_tokens

# modify the Qwen2 module to compute LRP in the backward pass
monkey_patch(modeling_qwen2, verbose=True)

# optional 4bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent overflow in gradients
)

path = '/content/Qwen/Qwen2.5-1.5B-Instruct'
model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(path, device_map='cuda', torch_dtype=torch.bfloat16, quantization_config=quantization_config)

# optional gradient checkpointing to save memory (2x forward pass)
model.train()
model.gradient_checkpointing_enable()

# deactive gradients on parameters to save memory
for param in model.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(path)

prompt = """Context: \
Louis Armstrong, born in 1901 in New Orleans, Louisiana, was a pioneering jazz trumpeter and vocalist known for his distinctive voice and virtuosic playing.\
His career spanned five decades, during which he revolutionized jazz music with hits like "What a Wonderful World."\
Armstrong's charisma and improvisational talent cemented his legacy as one of the greatest musicians in American history.\
In 1969, the first human walked on the Moon as part of the Apollo 11 mission, a significant achievement in space exploration.\
The mission was led by astronauts Neil Armstrong, Edwin "Buzz" Aldrin, and Michael Collins.\
Armstrong's famous words upon landing were, "That's one small step for man, one giant leap for mankind."\
Lance Armstrong, born in 1971, is a former professional cyclist who won the Tour de France seven consecutive times from 1999 to 2005.\
Though celebrated for his achievements, his career later became controversial due to doping allegations.\
Question: Who is the most famous person in the history of the Moon landing?\
Answer: Neil Armstrong is considered to be the most famous person in the history of the Moon landing. His iconic quote, "That's one small step for man, one giant leap for mankind," became an instant classic and is still remembered today. """

# get input embeddings so that we can compute gradients w.r.t. input embeddings
input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
input_embeds = model.get_input_embeddings()(input_ids)

# inference and get the maximum logit at the last position (we can also explain other tokens)
output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
# sum_logits, _indices = torch.max(output_logits[0, -1, :], dim=-1)

# 编码目标文本并获取 token ID（不加特殊符号）
# target_text = """ Neil Armstrong is considered to be the most famous person in the history of the Moon landing. His iconic quote, "That's one small step for man, one giant leap for mankind," became an instant classic and is still remembered today. """
target_text = """ Neil Armstrong is considered to be the most famous person in the history of the Moon landing. His iconic quote, "That's one small step for man, one giant leap for mankind," became an instant classic and is still remembered today. """
target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

# 将整段 prompt 的 input_ids 转换为列表
input_id_list = input_ids[0].tolist()

def find_subsequence(subseq, seq):
    for i in range(len(seq) - len(subseq), -1, -1):
        if seq[i:i+len(subseq)] == subseq:
            return i, i + len(subseq)
    return None, None

# 定位目标子序列的位置（在整个 prompt 中的 token 起止索引）
start_idx, end_idx = find_subsequence(target_ids, input_id_list)

# 异常处理：找不到目标子串
if start_idx is None:
    decoded_input = tokenizer.decode(input_id_list)
    raise ValueError(f"Target text tokens not found in input. Decoded input:\n{decoded_input}\nTarget:\n{target_text}")

print(f"Target tokens found at positions {start_idx} to {end_idx}")


# Get logits for the relevant token positions (excluding the last token for next-token prediction)
target_logits = output_logits[0, start_idx:end_idx]  # shape: [target_len, vocab_size]

# For each position, take the max logit value
max_logits_per_token = torch.max(target_logits, dim=-1).values  # shape: [target_len]

# Sum them
sum_max_logits = max_logits_per_token.sum()



# Backward pass (the relevance is initialized with the value of sum_max_logits)
# This initiates the LRP computation through the network
sum_max_logits.backward()

# obtain relevance by computing Gradient * Input
relevance = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0] # cast to float32 before summation for higher precision

# normalize relevance between [-1, 1] for plotting
relevance = relevance / relevance.abs().max()

# remove special characters from token strings and plot the heatmap
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
tokens = clean_tokens(tokens)

pdf_heatmap(tokens, relevance, path='/content/qwen2.5_1.5B_instruct_heatmap_sentence.pdf', backend='pdflatex') # backend='xelatex' supports more characters
