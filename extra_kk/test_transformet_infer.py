import transformers
import torch

model_id = "/home/kaifeng/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=0,
)

messages = [
    {
        "role": "system",
        "content": "你是一个智能助手，如实回答我提问的所有问题"
    },
    {
        "role": "user",
        "content": "/no_think你是谁"
    },
]

prompt = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Use only the <|eot_id|> as termination token
terminator = pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminator,  # Pass a single integer, not a list
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

print(outputs[0]["generated_text"][len(prompt):])