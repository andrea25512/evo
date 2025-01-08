import subprocess
import re

# Define parameter combinations
configs = [
    " --layers 3 --device cuda:1 --version long --coco --N 10 --learning_rate 0.00005",
    " --layers 3 --device cuda:1 --version long --coco --N 10 --learning_rate 0.00001"
]

# Iterate over each configuration and execute the script sequentially
for config in configs:
    # Create a unique log file name based on the parameters
    log_file = "run_" + config.replace(" ", "_").replace("=", "_") + ".out"
    log_file = re.sub(r'[^\w\-]', '_', log_file)
    
    # Build the full command
    command = f"python3 main.py {config}"
    
    print(f"Running: {command}")
    
    # Open the log file
    with open(log_file, "w") as log:
        # Run the command and wait for it to finish
        subprocess.run(
            command,
            shell=True,  # Allows string-based commands like in the shell
            stdout=log,  # Redirect standard output to the log file
            stderr=subprocess.STDOUT  # Redirect standard error to the same log file
        )