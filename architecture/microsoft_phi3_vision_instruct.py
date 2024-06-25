from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from vpt_llm.utils.prompting import load_frame_grid

model_id = "microsoft/Phi-3-vision-128k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="flash_attention_2",
)  # use _attn_implementation='eager' to disable flash attention

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": (
            "<|image_1|>\nDescribe what the player is doing / what actions "
            "they take in this clip of Minecraft gameplay. Respond like you "
            "are giving the player an instruction."
        ),
    },
]

image = load_frame_grid(
    "data/contractorV2/v10/00000",
    start_t=100,
    interval=10,
    height=4,
    width=3,
    correct_gamma=False,
)
image = Image.fromarray(image)

prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

generation_args = {
    "max_new_tokens": 128,
    "do_sample": False,
    # "temperature": 0.0,
}

generate_ids = model.generate(
    **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
)

# remove input tokens
generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(response)
