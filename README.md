# Evolutionary Prompt Tuning

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

4. Download the weights for `alpaca` (27 GB): https://drive.google.com/file/d/10EF-DaHyO1dN_hkH2J3IalzoA2j5UcD2/view.

5. Create a folder `weights` and extract the contents of the zip into it.
