from torch.utils.data import Dataset
import os
import pandas
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torch.utils.data import DataLoader, random_split
import torch
import transformers

script_dir = os.path.dirname(os.path.abspath(__file__))
device = "cuda:0"
seed = 42
test_images_number = 100

template = { 
    "standard": """
    Please follow the instruction step-by-step to generate a better prompt.  
    1. Crossover the following prompts to generate a new prompt:  
    Prompt 1: A photo of the small <tag>.
    Prompt 2: The <tag> in a video game.
    2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

    1. Crossover Prompt: A photo of the small <tag> in a video game.
    2. <prompt>A photo of the big <tag> in a video game.</prompt>

    Please follow the instruction step-by-step to generate a better prompt.
    1. Crossover the following prompts and generate a new prompt:
    Prompt 1: <prompt1>
    Prompt 2: <prompt2>
    2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.
    """           
}

class ImageDataset(Dataset):
    def __init__(self, images_root, csv_location, processor):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

        self.image_files = []
        for root, _, files in os.walk(os.path.join(script_dir, images_root)):
            for file in files:
                if file.lower().endswith(image_extensions):
                    self.image_files.append(os.path.join(root, file))

        self.map = pandas.read_csv(os.path.join(script_dir, csv_location))
        self.processor = processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = self.processor(images=Image.open(self.image_files[index]).convert('RGB'), return_tensors="pt")['pixel_values'][0]
        label = self.map[self.map.id == self.image_files[index].split("/")[-2]].description.item()
        return (image, label)

def evaluate(loader, query, clip_model, clip_processor):
    similarity = 0
    for images, labels in tqdm(loader):
        text_inputs = torch.stack([clip_processor(text=query.replace("<tag>", label), return_tensors="pt", padding=True)['input_ids'][0] for label in labels])
        similarity += clip_model(pixel_values=images.to(device), input_ids=text_inputs.to(device)).logits_per_image[0].cpu().detach().numpy()
        del images, text_inputs
        torch.cuda.empty_cache()
    similarity = similarity / len(loader)
    return similarity

def get_final_prompt(text):
    parts = text.split("<prompt>")
    if len(parts) > 1:
        prompt = parts[-1].split("</prompt>")[0]
        prompt = prompt.strip()
        return prompt
    else:
        if text.startswith("\"") and text.endswith("\""):
            text = text[1:-1]
        return text

def crossover_mutation(model, tokenizer, text1, text2):
    first_device = next(model.parameters()).device
    request_content = template["standard"].replace("<prompt1>", text1).replace("<prompt2>", text2)
    inputs = tokenizer(request_content, return_tensors="pt").to(first_device)
    out = model.generate(inputs=inputs.input_ids, max_new_tokens=100)
    output_text = tokenizer.batch_decode(out.cpu(), skip_special_tokens=True)[0]
    return get_final_prompt(output_text)

if __name__ == "__main__":
    torch.manual_seed(seed)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = clip_model.to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    alpaca_model = transformers.AutoModelForCausalLM.from_pretrained(os.path.join(script_dir,"weights/alpaca/"), device_map="auto", torch_dtype=torch.float32)
    alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained(os.path.join(script_dir,"weights/alpaca/"))

    prompt = crossover_mutation(alpaca_model, alpaca_tokenizer, "A fantasy illustration of a <tag>", "A sci-fi diagram involving a <tag>")
    print(prompt)

    del alpaca_model, alpaca_tokenizer
    torch.cuda.empty_cache()

    dataset = ImageDataset("data/imagenet-a","classes.csv", clip_processor)
    test_samples, _ = random_split(dataset, [test_images_number, len(dataset) - test_images_number])
    loader = DataLoader(test_samples, batch_size=1, shuffle=False, num_workers=1)

    similarity = evaluate(loader, prompt, clip_model, clip_processor)
    print(similarity)
    