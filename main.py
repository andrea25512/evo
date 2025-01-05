from torch.utils.data import Dataset
import os
import random
import pandas
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, BitsAndBytesConfig
from PIL import Image
from torch.utils.data import DataLoader, random_split
import torch
import transformers
import numpy as np
from pathlib import Path

script_dir = os.path.abspath(os.path.dirname(__file__))
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

        # Collect paths to all valid image files
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
        # Process the image
        image = self.processor(
            images=Image.open(self.image_files[index]).convert('RGB'),
            return_tensors="pt"
        )['pixel_values'][0]

        # Extract and normalize the file ID
        file_id = self.image_files[index].split("/")[-2].strip().lower()
        self.map["id"] = self.map["id"].str.strip().str.lower()

        # Find the matching description
        matches = self.map[self.map.id == file_id]
        if not matches.empty:
            label = matches.description.iloc[0]
        else:
            label = "Unknown"  # Handle unmatched cases

        return (image, label)


# Compute average similarity between images and caption templates
def evaluate(loader, query, clip_model, clip_processor):
    similarity = 0
    for images, labels in tqdm(loader):
        text_inputs = torch.stack([clip_processor(text=query.replace("<tag>", label), return_tensors="pt", padding=True)['input_ids'][0] for label in labels])
        similarity += clip_model(pixel_values=images.to(device), input_ids=text_inputs.to(device)).logits_per_image[0].cpu().detach().numpy()
        del images, text_inputs
        torch.cuda.empty_cache()
    similarity = similarity / len(loader)
    return similarity

# Extract the final prompt wrapped in <prompt> tags
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

# Generate a new prompt by combining two templates via an LLM
def crossover_mutation(model, tokenizer, text1, text2):
    first_device = next(model.parameters()).device
    request_content = template["standard"].replace("<prompt1>", text1).replace("<prompt2>", text2)
    inputs = tokenizer(request_content, return_tensors="pt").to(first_device)
    out = model.generate(inputs=inputs.input_ids, max_new_tokens=100)
    output_text = tokenizer.batch_decode(out.cpu(), skip_special_tokens=True)[0]
    return get_final_prompt(output_text)

# Tune prompts using GA
def ga_run(loader, initial_population, clip_model, clip_processor, model, tokenizer, generations=10, pop_size=10):
    population = initial_population
    best_prompt = None
    best_score = -float('inf')

    # Evaluate the fitness of the initial population
    fitness_scores = [float(evaluate(loader, prompt, clip_model, clip_processor)) for prompt in population]

    for generation in range(generations):
        print(f"\n=== Generation {generation + 1}/{generations} ===")

        # Track the best prompt
        max_score = max(fitness_scores)
        if max_score > best_score:
            best_score = max_score
            best_prompt = population[fitness_scores.index(max_score)]

        print(f"Best Score in Generation {generation + 1}: {max_score}")

        # Print the current population and fitness scores
        print("Population and Fitness Scores:")
        for i, (prompt, score) in enumerate(zip(population, fitness_scores)):
            print(f"  {i + 1}. {prompt} -> Fitness: {score:.4f}")

        # Selection using roulette wheel
        fitness_probs = np.array([score / sum(fitness_scores) for score in fitness_scores], dtype=np.float32)
        selected_parents = [
            population[i.item()] for i in torch.multinomial(
                torch.tensor(fitness_probs), num_samples=pop_size, replacement=True
            )
        ]

        # Crossover and mutation to generate children
        new_population = []
        for _ in range(pop_size):
            parent1, parent2 = random.sample(selected_parents, 2)
            child = crossover_mutation(model, tokenizer, parent1, parent2)
            new_population.append(child)

        # Compute fitness scores only for the new population
        new_fitness_scores = [
            float(evaluate(loader, prompt, clip_model, clip_processor)) for prompt in new_population
        ]

        # Combine the old population and new children
        combined_population = population + new_population
        combined_fitness_scores = fitness_scores + new_fitness_scores

        # Sort by fitness scores and retain the top N individuals
        sorted_indices = sorted(range(len(combined_fitness_scores)), key=lambda i: combined_fitness_scores[i], reverse=True)
        population = [combined_population[i] for i in sorted_indices[:pop_size]]
        fitness_scores = [combined_fitness_scores[i] for i in sorted_indices[:pop_size]]

    return best_prompt, best_score


if __name__ == "__main__":
    # Set up the models and dataset
    torch.manual_seed(seed)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # Quantize alpaca
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    weights_dir = os.path.normpath(os.path.join(script_dir, "weights/alpaca/"))
    alpaca_model = transformers.AutoModelForCausalLM.from_pretrained(
        weights_dir, quantization_config=quantization_config,
    )
    print("Model directory: ", weights_dir)
    alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained(weights_dir)

    dataset = ImageDataset("data/imagenet-a", "classes.csv", clip_processor)
    test_samples, _ = random_split(dataset, [test_images_number, len(dataset) - test_images_number])
    loader = DataLoader(test_samples, batch_size=1, shuffle=False, num_workers=1)

    # Initial population of prompts
    initial_population = [
        "A photo of a <tag>.",
        "An artistic rendering of a <tag>.",
        "A close-up shot of a <tag>.",
        "A diagram depicting a <tag>.",
        "A <tag> in a natural setting.",
        "A fantasy illustration of a <tag>.",
        "A sci-fi diagram involving a <tag>.",
        "An image of a small <tag>.",
        "A realistic photo of a <tag>.",
        "A digital artwork of a <tag>."
    ]

    # Run the genetic algorithm
    best_prompt, best_score = ga_run(loader, initial_population, clip_model, clip_processor, alpaca_model, alpaca_tokenizer)

    print(f"Best Prompt: {best_prompt}")
    print(f"Best Score: {best_score}")
