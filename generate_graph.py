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

script_dir = os.path.abspath(os.path.dirname(__file__))

file_name = "de_generations_10_population_20_donor_random_True"

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
