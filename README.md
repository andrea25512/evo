# Evolutionary Prompt Tuning

Repository for the project of the course of *Bio-Inspired Artificial Intelligence* (Academic Year 2024/2025).

In this project we adapt [EvoPrompt](https://arxiv.org/abs/2309.08532), a framework for automatic prompt optimization, to the task of prompt optimization for vision-language models (e.g. [CLIP](https://github.com/openai/CLIP)). EvoPrompt works by connecting LLMs to Evolutionary Algorithms. In particular: Genetic Algorithm and Differential Evolution.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/evo.git
   cd evo
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   On Windows, you also need to run:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

3. Install `alpaca`:
   ```bash
   git clone https://github.com/tatsu-lab/stanford_alpaca.git
   ```

4. Download the weights for `alpaca` (27 GB): https://drive.google.com/file/d/10EF-DaHyO1dN_hkH2J3IalzoA2j5UcD2/view. Create a folder `weights` and extract the contents of the zip into it.

5. Download the datasets and extract them in a folder `data`:
   - Imagenet-A (https://drive.google.com/file/d/1O-Ljtr99F2JI4QW9KSO3DjipI3oRkfyY/view?usp=sharing)
   - Flower102 (https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
   - FGVC-Aircraft (https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)

## Experiments

To reproduce our experiments:
- Genetic Algorithm
```bash
python gen_alg_run.py
```

- Differential Evolution
```bash
python diff_evo_run.py
```

The default dataset is `imagenet-a`. To test other datasets (`Flowers102` and `FGVC-Aircraft`), add their corresponding flag:
```bash
python diff_evo_run.py --flowers
python diff_evo_run.py --air
```
## Code Structure
```bash
.
├── csv_recap # recaps of the runs
├── data/ #datasets
│   ├── imagenet-a
│   ├── flowers
│   └── planes
├── images # graphs of the runs
├── stanford_alpaca # LLM
├── weights # alpaca's weights
├── classes.csv # imagenet-a labels
├── diff_evo_run.py # script to run the diff_evo with different settings
├── diff_evo.py # main script for differential evolution
├── gen_alg_run.py # script to run the gen_alg with different settings
├── gen_alg.py # main script for genetic algorithm
├── generate_graph.py # util for generating graphs
├── labels.csv # Flowers102 labels
├── planes.csv # FGVC-Aircraft labels
└── requirements.txt
```

## Citation
If you find this work useful, please consider citing:
```bibtex
@InProceedings{10.1007/978-3-032-23607-4_11,
author="Ceron, Andrea
and Don{\`a}, Ivan
and Iacca, Giovanni",
editor="Garc{\'i}a-S{\'a}nchez, Pablo
and D{\'i}az {\'A}lvarez, Josefa
and Murphy, Aidan",
title="CLIP Prompt Optimization with Evolutionary-Driven LLMs",
booktitle="Applications of Evolutionary Computation",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="167--182",
isbn="978-3-032-23607-4"
}
```
