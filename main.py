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
import random
import argparse
import json
from difflib import SequenceMatcher

script_dir = os.path.abspath(os.path.dirname(__file__))
device = "cuda:0"
seed = 42
test_images_number = 100
os.environ["TOKENIZERS_PARALLELISM"] = "true"
seen_words = []
niche_threshold = 0.95

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
def crossover_mutation(model, tokenizer, text1, text2):
    first_device = next(model.parameters()).device
    request_content = template["standard"].replace("<prompt1>", text1).replace("<prompt2>", text2)
    inputs = tokenizer(request_content, return_tensors="pt").to(first_device)
    out = model.generate(inputs=inputs.input_ids, max_new_tokens=100)
    output_text = tokenizer.batch_decode(out.cpu(), skip_special_tokens=True)[0]
    return get_final_prompt(output_text)

# ====================================================
# SECTION: Parent selection strategies
# ====================================================

def roulette_wheel_selection(fitness_scores, population, children_number):
    # selection using roulette wheel
    fitness_probs = np.array([score / sum(fitness_scores) for score in fitness_scores], dtype=np.float32)
    selected_parents = [
        population[i.item()] for i in torch.multinomial(
            torch.tensor(fitness_probs), num_samples=children_number, replacement=True
        )
    ]
    return selected_parents

def tournament_selection(fitness_scores, population, children_number, tournament_size):
    selected_parents = []
    for _ in range(children_number):
        # randomly select `tournament_size` individuals from the population
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_individuals = [population[i] for i in tournament_indices]
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        # Select the individual with the highest fitness
        best_individual_index = np.argmax(tournament_fitness)
        selected_parents.append(tournament_individuals[best_individual_index])

    return selected_parents

def rank_selection(fitness_scores, population, children_number):
    # Rank individuals based on their fitness scores
    # Negative sign for descending order
    sorted_indices = np.argsort(-np.array(fitness_scores))  
    ranked_population = [population[i] for i in sorted_indices]
    
    # Assign ranks (highest fitness gets highest rank)
    ranks = np.arange(len(fitness_scores), 0, -1)

    # Assign probabilities proportional to ranks
    rank_sum = sum(ranks)
    rank_probs = np.array([rank / rank_sum for rank in ranks], dtype=np.float32)

    # Use roulette wheel or similar selection method
    selected_parents = [
        ranked_population[i.item()] for i in torch.multinomial(
            torch.tensor(rank_probs), num_samples=children_number, replacement=True
        )
    ]

    return selected_parents

def truncated_rank_selection(fitness_scores, population, children_number):
    # Rank individuals based on their fitness scores
    # Negative sign for descending order
    sorted_indices = np.argsort(-np.array(fitness_scores))
    ranked_population = [population[i] for i in sorted_indices]

    # Select the top N individuals as parents
    selected_parents = ranked_population[:children_number]

    return selected_parents
    
# ====================================================
# SECTION: Evolutionary algorithms
# ====================================================

# Tune prompts using GA
def ga_run(loader, initial_population, clip_model, clip_processor, model, tokenizer, generations=10, pop_size=50, children_number=100, selection_index=0, replacement_index=0, file_name="TEST"):
    last_average_length = 0
    population = initial_population
    best_prompt = None
    best_score = -float('inf')

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

        # Print the current population and fitness scores
        print(f"Best Score in Generation {generation + 1}: {max_score}")
        print("Population and Fitness Scores:")
        for i, (prompt, score) in enumerate(zip(population, fitness_scores)):
            print(f"  {i + 1}. {prompt} -> Fitness: {score:.4f}")
        print("\n")

        if(not generation + 1 == generations):
            if(selection_index == 0):
                selected_parents = roulette_wheel_selection(fitness_scores, population, children_number)
            elif(selection_index == 1):
                selected_parents = tournament_selection(fitness_scores, population, children_number, 2)
            elif(selection_index == 2):
                selected_parents = rank_selection(fitness_scores, population, children_number)
            else:
                raise ValueError("Wrong selection index")

            children=[]
            # Crossover and mutation to generate children
            for _ in tqdm(range(children_number), desc="Generation"):
                parent1, parent2 = random.sample(selected_parents, 2)
                child = crossover_mutation(model, tokenizer, parent1, parent2)
                children.append(child)

            # Compute fitness scores only for the new population
            children_fitness_scores = [get_fitness(loader, prompt, clip_model, clip_processor) for prompt in tqdm(children, desc="Evaluation")]

            # Combine the old population and new children
            if(replacement_index == 0):
                # (µ+λ)
                combined_population = population + children
                combined_fitness_scores = fitness_scores + children_fitness_scores

                # Sort by fitness scores and retain the top N individuals
                sorted_indices = sorted(range(len(combined_fitness_scores)), key=lambda i: combined_fitness_scores[i], reverse=True)
                population = [combined_population[i] for i in sorted_indices[:pop_size]]
                fitness_scores = [combined_fitness_scores[i] for i in sorted_indices[:pop_size]]
            elif(replacement_index == 1):
                # (µ,λ)
                combined_population = children
                combined_fitness_scores = children_fitness_scores

                # Sort by fitness scores and retain the top N individuals
                sorted_indices = sorted(range(len(combined_fitness_scores)), key=lambda i: combined_fitness_scores[i], reverse=True)
                population = [combined_population[i] for i in sorted_indices[:pop_size]]
                fitness_scores = [combined_fitness_scores[i] for i in sorted_indices[:pop_size]]
            elif(replacement_index == 2):
                combined_population = population + children
                combined_fitness_scores = fitness_scores + children_fitness_scores

                niches = []
                niches_fitness = []
                for (prompt, fitness) in zip(combined_population, combined_fitness_scores):
                    # check if the prompt fits into an existing niche
                    added_to_niche = False
                    max_sim = 0
                    for (niche, niche_fitness) in zip(niches, niches_fitness):
                        # compare with all prompts of the niche
                        sim = max([SequenceMatcher(None, prompt, niche_prompt).ratio() for niche_prompt in niche])
                        if(sim > max_sim):
                            max_sim = sim
                        if  sim > niche_threshold:
                            niche.append(prompt)
                            niche_fitness.append(fitness)
                            added_to_niche = True
                            break
                    # if no suitable niche exists, create a new one
                    if not added_to_niche:
                        print("New niche with: ", prompt, " - max similarity: ",max_sim)
                        niches.append([prompt])
                        niches_fitness.append([fitness])

                # calculate average fitness for each niche
                niche_avg_fitness = [sum(niche_fitness) / len(niche_fitness) if niche_fitness else 0 for niche_fitness in niches_fitness]

                # compute selection probabilities by normalizing average fitness
                total_avg_fitness = sum(niche_avg_fitness)
                if total_avg_fitness > 0:
                    niche_selection_probabilities = [avg_fitness / total_avg_fitness for avg_fitness in niche_avg_fitness]
                else:
                    # equal probability if all averages are zero
                    niche_selection_probabilities = [1 / len(niches) for _ in niches]  

                # determine how many samples to take from each niche
                niche_sample_counts = [int(round(pop_size * prob)) for prob in niche_selection_probabilities]
                
                # adjust sample counts to ensure the total matches pop_size
                while sum(niche_sample_counts) > pop_size:
                    # reduce from niches with the lowest selection probabilities
                    niche_sample_counts[niche_sample_counts.index(max(niche_sample_counts))] -= 1
                while sum(niche_sample_counts) < pop_size:
                    # add to niches with the highest selection probabilities
                    niche_sample_counts[niche_sample_counts.index(min(niche_sample_counts))] += 1

                # Compute selection probabilities by normalizing average fitness
                total_avg_fitness = sum(niche_avg_fitness)
                if total_avg_fitness > 0:
                    niche_selection_probabilities = [avg_fitness / total_avg_fitness for avg_fitness in niche_avg_fitness]
                else:
                    # equal probability if all averages are zero
                    niche_selection_probabilities = [1 / len(niches) for _ in niches]  

                # select prompts and fitness scores for the next generation
                population = []
                fitness_scores = []

                for niche, niche_fitness, sample_count in zip(niches, niches_fitness, niche_sample_counts):
                    # select the top 'sample_count' elements from each niche
                    if sample_count > 0:
                        top_indices = sorted(range(len(niche_fitness)), key=lambda i: niche_fitness[i], reverse=True)[:sample_count]
                        population.extend([niche[i] for i in top_indices])
                        fitness_scores.extend([niche_fitness[i] for i in top_indices])

                # sort the population and fitness_scores based on fitness
                sorted_pairs = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
                population, fitness_scores = zip(*sorted_pairs)
                population = list(population)
                fitness_scores = list(fitness_scores)

            else:
                raise ValueError("Wrong replacement index")

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
    parser.add_argument('-c', '--children', type=int, help='Children amount', default=20)
    parser.add_argument('-g', '--generations', type=int, help='Generation amount', default=10)
    parser.add_argument('-s', '--selection', type=int, help='Selection index', default=0)
    parser.add_argument('-r', '--replacement', type=int, help='Replacement index', default=0) # zero fixed for now as only elitism is avelable

    args = vars(parser.parse_args())

    generations = args['generations']
    pop_size = args['population']
    child_size = args['children']
    selection_index = args['selection']
    replacement_index = args['replacement']
    
    file_name = f"generations_{generations}_population_{pop_size}_children_{child_size}_selection_{selection_index}_replacement_{replacement_index}"
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

    initial_population = random.sample(initial_population, k=pop_size)

    # Run the genetic algorithm
    best_prompt, best_score = ga_run(loader, initial_population, clip_model, clip_processor, alpaca_model, alpaca_tokenizer, generations, pop_size, child_size, selection_index, replacement_index, file_name)

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
