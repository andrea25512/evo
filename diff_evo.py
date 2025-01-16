from torch.utils.data import Dataset
import os
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
import random
import argparse
import json

script_dir = os.path.abspath(os.path.dirname(__file__))
device = "cuda:0"
seed = 42
test_images_number = 100
os.environ["TOKENIZERS_PARALLELISM"] = "true"
seen_words = []

template = { 
    "compressed_differential_evolution": """### Instructions
    Follow these steps to improve a given prompt, without changing its structure:
    1. Identify differences between Prompt 1 and Prompt 2.
    2. Randomly mutate the different parts.
    3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt:
    4. Crossover the prompt in step 3 with the basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:

    ### Task
    Apply the steps to these prompts, without changing their structure:
    Prompt 1: <prompt1>  
    Prompt 2: <prompt2>  
    Prompt 3: <prompt3>  
    Basic Prompt: <prompt0>  

    Output: Final improved prompt enclosed in <prompt> and </prompt>.
    """,
    "differential_evolution": """Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: A photo of the small <tag>.
Prompt 2: a cropped photo of the <tag>.
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt.
Prompt 3: a low resolution photo of the <tag>.
4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: a close-up photo of the <tag>.

1. Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: A photo of the small <tag>.
Prompt 2: a cropped photo of the <tag>.
Different parts:
"A photo of the small" vs "a cropped photo of the"

2. Randomly mutate the different parts:
"A photo of the small" -> "An image of the tiny" 
"a cropped photo of the" -> "a cut out photo of the"

3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt:
Prompt 3: a low resolution photo of the <tag>.
New Prompt: A cut out image of the tiny <tag>.

4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: a close-up photo of the <tag>.
Final Prompt: <prompt>A close-up cut out image of the tiny <tag>.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step2 and generate a new prompt.
Prompt 3: <prompt3>
4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: <prompt0>

1. """
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

# ====================================================
# SECTION: Utility functions
# ====================================================

# Compute average similarity between images and caption templates
def get_fitness(loader, prompt, clip_model, clip_processor):
    similarity = 0
    for images, labels in loader:
        text_inputs = clip_processor(text=[prompt.replace("<tag>", label) for label in labels], return_tensors="pt", padding=True)['input_ids']
        similarity += np.sum(clip_model(pixel_values=images.to(device), input_ids=text_inputs.to(device)).logits_per_image.cpu().detach().numpy())
        del images, text_inputs
        torch.cuda.empty_cache()
    similarity = similarity / test_images_number
    
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

# ====================================================
# SECTION: Genetic operators
# ====================================================

# Generate a new prompt by combining two templates via an LLM
def crossover_mutation(model, tokenizer, ea_strategy, text1, text2, text3=None, text4=None):
    first_device = next(model.parameters()).device
    if ea_strategy == "genetic_algorithm":
        request_content = template[ea_strategy].replace("<prompt1>", text1).replace("<prompt2>", text2)
    elif ea_strategy == "differential_evolution":
        request_content = template[ea_strategy].replace("<prompt0>", text1).replace("<prompt1>", text2).replace("<prompt2>", text3).replace("<prompt3>", text4)
    else:
        raise ValueError("Invalid evolutionary algorithm strategy. Use 'genetic_algorithm' or 'differential_evolution'.")
    inputs = tokenizer(request_content, return_tensors="pt", truncation=True).to(first_device)
    out = model.generate(inputs=inputs.input_ids, max_new_tokens=512)
    output_text = tokenizer.batch_decode(out.cpu(), skip_special_tokens=True)[0]
    print("Mutant: ", get_final_prompt(output_text))
    return get_final_prompt(output_text)
    

# ====================================================
# SECTION: Evolutionary algorithms
# ====================================================

def de_run(loader, initial_population, clip_model, clip_processor, model, tokenizer, generations=10, pop_size=50, donor_random=False, file_name="TEST"):
    last_average_length = 0
    population = initial_population
    best_prompt = None
    best_score = -float('inf')
    ea_strategy = "differential_evolution"
    local_random = random.Random()

    # DOES NOT WORK
    """ # Define the file path for saving fitness scores
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
        print("Saved initial fitness scores to file.") """
        
    # Evaluate the fitness of the initial population
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
        'timestamp': [0],
        'best_fitness': [best_fitness],
        'average_fitness': [average_fitness],
        'worst_fitness': [worst_fitness],
        'average_length': [average_length],
        'variance_length': [variance_length],
        'added_word': [added_word]
    })
    tab_metrics.to_csv(os.path.join(script_dir, f"csv_recap/{file_name}.csv"), mode='a', header=False, index=False)

    for generation in range(generations):
        print(f"=== Generation {generation + 1}/{generations} ===")

        # Track the best prompt
        max_score = max(fitness_scores)
        if max_score > best_score:
            best_score = max_score
            best_prompt = population[fitness_scores.index(max_score)]

        # re-ordering based on the fitness
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]

        # Print the current population and fitness scores
        print(f"Best Score in Generation {generation + 1}: {max_score}")
        print("Population and Fitness Scores:")
        for i, (prompt, score) in enumerate(zip(population, fitness_scores)):
            print(f"  {i + 1}. {prompt} -> Fitness: {score:.4f}")
        print("\n")

        # Crossover and mutation to generate children
        for i in tqdm(range(pop_size), desc="Generation"):
            local_random.seed(i)
            # select each individual
            candidate = population[i]
            # select randomly three samples from the list, where the candidate is already excluded
            donor1, donor2, donor3 = local_random.sample([x for x in population if x != candidate], 3)
            # the third sampling is not needed if we use the current best prompt as value to which to add the F(rand_1 - rand_2)
            if not donor_random:
                donor3 = best_prompt
            else:
                print("donor_random")
            print(f"Original prompt: {candidate} \nDonors: {donor1}, {donor2}, {donor3}")
            mutant = crossover_mutation(model, tokenizer, ea_strategy, candidate, donor1, donor2, donor3)
            mutant_score = get_fitness(loader, mutant, clip_model, clip_processor)

            # Replace the candidate with the mutant if it has a higher fitness score
            if mutant_score > fitness_scores[population.index(candidate)]:
                population[i] = mutant
                fitness_scores[i] = mutant_score

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
            'timestamp': [generation + 1],
            'best_fitness': [best_fitness],
            'average_fitness': [average_fitness],
            'worst_fitness': [worst_fitness],
            'average_length': [average_length],
            'variance_length': [variance_length],
            'added_word': [added_word]
        })
        tab_metrics.to_csv(os.path.join(script_dir, f"csv_recap/{file_name}.csv"), mode='a', header=False, index=False)

    return best_prompt, best_score


# ====================================================
# SECTION: Main script
# ====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intercept command-line parameters.")
    parser.add_argument('-p', '--population', type=int, help='Population amount', default=10)
    parser.add_argument('-g', '--generations', type=int, help='Generation amount', default=10)
    parser.add_argument('-d', '--donor_random', action='store_true', help='Enable random donor')

    args = vars(parser.parse_args())

    generations = args['generations']
    pop_size = args['population']
    if args['donor_random']:
        donor_random = True
    else:
        donor_random = False
    
    file_name = f"de_generations_{generations}_population_{pop_size}_donor_random_{donor_random}"
    print(f"Output file name: {file_name}")

    # Set up the models and dataset
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # Quantization settings
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    weights_dir = os.path.normpath(os.path.join(script_dir, "weights/alpaca/"))
    #alpaca_model = transformers.AutoModelForCausalLM.from_pretrained(weights_dir, quantization_config=quantization_config)
    alpaca_model = transformers.AutoModelForCausalLM.from_pretrained(os.path.join(script_dir,"weights/alpaca/"), device_map="auto")
    print("Model directory: ", weights_dir)
    alpaca_tokenizer = transformers.AutoTokenizer.from_pretrained(weights_dir)

    dataset = ImageDataset("data/imagenet-a", "classes.csv", clip_processor)
    test_samples, _ = random_split(dataset, [test_images_number, len(dataset) - test_images_number])
    # LEAVE THE BATCH SIZE AND NUMBER OF WORKERS TO 1!!!!!!!!!!!
    loader = DataLoader(test_samples, batch_size=1, shuffle=False, num_workers=1)

    # Initial population of prompts
    initial_population = [
        'a bad photo of a <tag>.',
        'a photo of many <tag>.',
        'a sculpture of a <tag>.',
        'a photo of the hard to see <tag>.',
        'a low resolution photo of the <tag>.',
        'a rendering of a <tag>.',
        'graffiti of a <tag>.',
        'a bad photo of the <tag>.',
        'a cropped photo of the <tag>.',
        'a tattoo of a <tag>.',
        'the embroidered <tag>.',
        'a photo of a hard to see <tag>.',
        'a bright photo of a <tag>.',
        'a photo of a clean <tag>.',
        'a photo of a dirty <tag>.',
        'a dark photo of the <tag>.',
        'a drawing of a <tag>.',
        'a photo of my <tag>.',
        'the plastic <tag>.',
        'a photo of the cool <tag>.',
        'a close-up photo of a <tag>.',
        'a black and white photo of the <tag>.',
        'a painting of the <tag>.',
        'a painting of a <tag>.',
        'a pixelated photo of the <tag>.',
        'a sculpture of the <tag>.',
        'a bright photo of the <tag>.',
        'a cropped photo of a <tag>.',
        'a plastic <tag>.',
        'a photo of the dirty <tag>.',
        'a jpeg corrupted photo of a <tag>.',
        'a blurry photo of the <tag>.',
        'a photo of the <tag>.',
        'a good photo of the <tag>.',
        'a rendering of the <tag>.',
        'a <tag> in a video game.',
        'a photo of one <tag>.',
        'a doodle of a <tag>.',
        'a close-up photo of the <tag>.',
        'a photo of a <tag>.',
        'the origami <tag>.',
        'the <tag> in a video game.',
        'a sketch of a <tag>.',
        'a doodle of the <tag>.',
        'a origami <tag>.',
        'a low resolution photo of a <tag>.',
        'the toy <tag>.',
        'a rendition of the <tag>.',
        'a photo of the clean <tag>.',
        'a photo of a large <tag>.',
        'a rendition of a <tag>.',
        'a photo of a nice <tag>.',
        'a photo of a weird <tag>.',
        'a blurry photo of a <tag>.',
        'a cartoon <tag>.',
        'art of a <tag>.',
        'a sketch of the <tag>.',
        'a embroidered <tag>.',
        'a pixelated photo of a <tag>.',
        'itap of the <tag>.',
        'a jpeg corrupted photo of the <tag>.',
        'a good photo of a <tag>.',
        'a plushie <tag>.',
        'a photo of the nice <tag>.',
        'a photo of the small <tag>.',
        'a photo of the weird <tag>.',
        'the cartoon <tag>.',
        'art of the <tag>.',
        'a drawing of the <tag>.',
        'a photo of the large <tag>.',
        'a black and white photo of a <tag>.',
        'the plushie <tag>.',
        'a dark photo of a <tag>.',
        'itap of a <tag>.',
        'graffiti of the <tag>.',
        'a toy <tag>.',
        'itap of my <tag>.',
        'a photo of a cool <tag>.',
        'a photo of a small <tag>.',
        'a tattoo of the <tag>.',
    ]

    extended_population = [
        "a vintage photo of the <tag>.",
        "a hyper-realistic painting of a <tag>.",
        "a minimalist sketch of the <tag>.",
        "a surreal rendering of a <tag>.",
        "an abstract painting of the <tag>.",
        "a futuristic version of a <tag>.",
        "a steampunk-style <tag>.",
        "a photo of a broken <tag>.",
        "a 3D printed <tag>.",
        "a hand-drawn comic of the <tag>.",
        "a distorted photo of a <tag>.",
        "a high contrast photo of the <tag>.",
        "a holographic image of a <tag>.",
        "a stop-motion model of the <tag>.",
        "an anime-style drawing of the <tag>.",
        "a watercolor painting of the <tag>.",
        "a cyberpunk depiction of a <tag>.",
        "a clay sculpture of the <tag>.",
        "a digital collage featuring a <tag>.",
        "a neon sign shaped like the <tag>."
    ]

    if(pop_size > 80):
        initial_population = initial_population + extended_population
    initial_population = random.sample(initial_population, k=pop_size)

    # Run the genetic algorithm
    best_prompt, best_score = de_run(loader, initial_population, clip_model, clip_processor, alpaca_model, alpaca_tokenizer, generations, pop_size, donor_random=donor_random, file_name=file_name)
    #mutant = crossover_mutation(alpaca_model, alpaca_tokenizer, "differential_evolution", 'a jpeg corrupted photo of the <tag>.', 'a photo of a nice <tag>.', 'a rendition of the <tag>.', 'a bright photo of a <tag>.')
    #print(mutant)
    print(f"Best Prompt: {best_prompt}")
    print(f"Best Score: {best_score}")

    data = pandas.read_csv(os.path.join(script_dir, f"csv_recap/{file_name}.csv"), header=None)

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
    plt.savefig(os.path.join(script_dir, f"images/{file_name}.png"))
