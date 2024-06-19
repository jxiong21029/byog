import time

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)

from vpt_llm.utils.prompting import load_frame_grid

start_time = time.time()
model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
model = (
    LlavaNextForConditionalGeneration
    if "1.6" in model_name
    else LlavaForConditionalGeneration
).from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)
print(f"loading time: {time.time() - start_time:.2f}")

start_time = time.time()
# Load frame grid
image1 = load_frame_grid(
    "data/contractorV2/v10/00000",
    start_t=100,
    interval=10,
    height=4,
    width=3,
    correct_gamma=False,
)
image2 = load_frame_grid(
    "data/contractorV2/v10/00000",
    start_t=200,
    interval=10,
    height=4,
    width=3,
    correct_gamma=False,
)
images = [Image.fromarray(image1), Image.fromarray(image2)]
images[0].save("byog/image1.png")
images[1].save("byog/image2.png")

while True:
    if "1.6" in model_name:
        with open("byog/prompt-v16.txt", "r") as f:
            prompt = " ".join(line.strip() for line in f).strip()
    elif "1.5" in model_name:
        with open("byog/prompt-v15.txt", "r") as f:
            prompt = " ".join(line.strip() for line in f).strip()
    else:
        raise ValueError
    prompt = prompt.replace("\\n", "\n")
    print(f"{prompt=}")
    # prompts = [
    #     "USER: <image>\nDescribe what the player is doing / what actions they take in this clip of Minecraft gameplay. "
    #     "Respond with one sentence in imperative tense, like you are giving the player an instruction. "
    #     "Do not mention frame numbers or the name of the game. ASSISTANT:",
    # ] * 2
    prompts = [prompt] * len(images)
    inputs = processor(text=prompts, images=images, return_tensors="pt").to(0)
    print(f"preprocessing time: {time.time() - start_time:.2f}")

    start_time = time.time()
    generate_ids = model.generate(**inputs, max_new_tokens=50)
    outputs = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(f"generation time: {time.time() - start_time:.2f}")

    for prompt, output in zip(prompts, outputs):
        print(output)

    cmd = input("continue? (y/n): ")
    if not cmd.startswith("y"):
        break
