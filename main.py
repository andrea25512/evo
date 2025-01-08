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
import pandas
import string
import matplotlib.pyplot as plt
import json

script_dir = os.path.abspath(os.path.dirname(__file__))
device = "cuda:0"
seed = 42
test_images_number = 100
os.environ["TOKENIZERS_PARALLELISM"] = "true"
seen_words = []


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
def get_fitness(loader, prompt, clip_model, clip_processor):
    similarity = 0
    for images, labels in loader:
        text_inputs = torch.stack([clip_processor(text=prompt.replace("<tag>", label), return_tensors="pt", padding=True)['input_ids'][0] for label in labels])
        similarity += clip_model(pixel_values=images.to(device), input_ids=text_inputs.to(device)).logits_per_image[0].cpu().detach().numpy()
        del images, text_inputs
        torch.cuda.empty_cache()
    similarity = similarity[0] / len(loader)
    
    return similarity

def evaluate(loader, population, clip_model, clip_processor, timestamp, last_average_length):
    fitness_scores = [get_fitness(loader, prompt, clip_model, clip_processor) for prompt in tqdm(population, desc="Evaluation")]
    best_fitness = np.max(fitness_scores)
    average_fitness = np.average(fitness_scores)
    worst_fitness = np.min(fitness_scores)
    average_length = np.average([len(prompt) for prompt in population])
    variance_length = average_length - last_average_length
    last_average_length = average_length
    added_word = 0
    for prompt in population:
        for word in prompt.split(" "):
            word = word.strip().lower().translate(str.maketrans('', '', string.punctuation))
            if(word not in seen_words):
                added_word += 1
                seen_words.append(word)

    tab_metrics = pandas.DataFrame({
        'timestamp': [timestamp],
        'best_fitness': [best_fitness],
        'average_fitness': [average_fitness],
        'worst_fitness': [worst_fitness],
        'average_length': [average_length],
        'variance_length': [variance_length],
        'added_word': [added_word]
    })
    tab_metrics.to_csv(os.path.join(script_dir, "run_recap.csv"), mode='a', header=False, index=False)

    return (fitness_scores, last_average_length)

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

def load_fitness_scores(file_path, to_numpy_float32=False):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r") as f:
            scores = json.load(f)
            if to_numpy_float32:
                # Convert Python float to numpy.float32
                scores = [np.float32(score) for score in scores]
            return scores
    return None

def save_fitness_scores(file_path, fitness_scores):
    # Convert numpy.float32 to float to ensure JSON compatibility
    fitness_scores = [float(score) for score in fitness_scores]
    with open(file_path, "w") as f:
        json.dump(fitness_scores, f)


# Tune prompts using GA
def ga_run(loader, initial_population, clip_model, clip_processor, model, tokenizer, generations=10, pop_size=10, children_number=20):
    last_average_length = 0
    population = initial_population
    best_prompt = None
    best_score = -float('inf')

    # Define the file path for saving fitness scores
    fitness_file = "fitness_scores.json"

    # Attempt to load fitness scores from the file
    saved_fitness_scores = load_fitness_scores(fitness_file)

    if saved_fitness_scores is not None:
        # If scores exist, use them
        fitness_scores = saved_fitness_scores
        print("Loaded fitness scores from file.")
    else:
        # If no saved scores, evaluate the fitness of the initial population
        fitness_scores, last_average_length = evaluate(loader, population, clip_model, clip_processor, 0, last_average_length)
        save_fitness_scores(fitness_file, fitness_scores)  # Save the computed scores to the file
        print("Saved initial fitness scores to file.")



    for generation in range(generations):
        print(f"=== Generation {generation + 1}/{generations} ===")

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
        print("\n")

        if(not generation + 1 == generations):
            # Selection using roulette wheel
            fitness_probs = np.array([score / sum(fitness_scores) for score in fitness_scores], dtype=np.float32)
            selected_parents = [
                population[i.item()] for i in torch.multinomial(
                    torch.tensor(fitness_probs), num_samples=children_number, replacement=True
                )
            ]
            # Crossover and mutation to generate children
            new_population = []
            for _ in tqdm(range(children_number), desc="Generation"):
                parent1, parent2 = random.sample(selected_parents, 2)
                child = crossover_mutation(model, tokenizer, parent1, parent2)
                new_population.append(child)

            # Compute fitness scores only for the new population
            (new_fitness_scores, last_average_length) = evaluate(loader, new_population, clip_model, clip_processor, generation + 1, last_average_length)

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
    # Quantization settings
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    weights_dir = os.path.normpath(os.path.join(script_dir, "weights/alpaca/"))
    alpaca_model = transformers.AutoModelForCausalLM.from_pretrained(weights_dir, quantization_config=quantization_config)
    #alpaca_model = transformers.AutoModelForCausalLM.from_pretrained(os.path.join(script_dir,"weights/alpaca/"), device_map="auto")
    print("Model directory: ", weights_dir)
    alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained(weights_dir)

    dataset = ImageDataset("data/imagenet-a", "classes.csv", clip_processor)
    test_samples, _ = random_split(dataset, [test_images_number, len(dataset) - test_images_number])
    loader = DataLoader(test_samples, batch_size=1, shuffle=False, num_workers=1)

    # Initial population of prompts
    initial_population = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]

    # Run the genetic algorithm
    best_prompt, best_score = ga_run(loader, initial_population, clip_model, clip_processor, alpaca_model, alpaca_tokenizer)

    print(f"Best Prompt: {best_prompt}")
    print(f"Best Score: {best_score}")

    data = pandas.read_csv(os.path.join(script_dir, "run_recap.csv"), header=None)

    # Assign column names based on the description
    data.columns = ["timestamp", "best_fitness", "average_fitness", "worst_fitness", "average_length", "variance_length", "added_word"]

    # Create a 2x2 grid for the plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Variation in time of best, average, and worst fitness
    axes[0, 0].plot(data["timestamp"], data["best_fitness"], label="Best Fitness")
    axes[0, 0].plot(data["timestamp"], data["average_fitness"], label="Average Fitness")
    axes[0, 0].plot(data["timestamp"], data["worst_fitness"], label="Worst Fitness")
    axes[0, 0].set_xlabel("Timestamp")
    axes[0, 0].set_ylabel("Fitness")
    axes[0, 0].set_title("Fitness Variation Over Time")
    axes[0, 0].legend()
    axes[0, 0].grid()

    # Plot 2: Variation in time of average length
    axes[0, 1].plot(data["timestamp"], data["average_length"], label="Average Length", color="orange")
    axes[0, 1].set_xlabel("Timestamp")
    axes[0, 1].set_ylabel("Average Length")
    axes[0, 1].set_title("Average Length Over Time")
    axes[0, 1].grid()

    # Plot 3: Variation in time of variance length
    axes[1, 0].plot(data["timestamp"], data["variance_length"], label="Variance Length", color="green")
    axes[1, 0].set_xlabel("Timestamp")
    axes[1, 0].set_ylabel("Variance Length")
    axes[1, 0].set_title("Variance Length Over Time")
    axes[1, 0].grid()

    # Plot 4: Variation in time of added words
    axes[1, 1].plot(data["timestamp"], data["added_word"], label="Added Words", color="red")
    axes[1, 1].set_xlabel("Timestamp")
    axes[1, 1].set_ylabel("Added Words")
    axes[1, 1].set_title("Added Words Over Time")
    axes[1, 1].grid()

    # Save the combined figure
    plt.savefig(os.path.join(script_dir, "combined_graphs.png"))
