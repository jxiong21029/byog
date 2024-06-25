from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from vpt_llm.utils.prompting import load_frame_grid

# class Florence2Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()


model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large-ft", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large-ft", trust_remote_code=True
)
print(f"param count: {sum(p.numel() for p in model.parameters()):,}")

image = load_frame_grid(
    "data/contractorV2/v10/00000",
    start_t=100,
    interval=20,
    height=2,
    width=2,
    correct_gamma=False,
    add_labels=False,  # NOTE: turn off here since caption model, not instruct
)
image = Image.fromarray(image)
image.save("byog/data/image1a.png")

for prompt in ("<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"):
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=128,
        do_sample=False,
    )
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )
    print(parsed_answer)
