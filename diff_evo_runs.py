import subprocess
import re

# Define parameter combinations
configs = [
    "--generations 10 --population 10 donor_random True",
    "--generations 10 --population 10 donor_random False",
    "--generations 10 --population 20 donor_random True",
    "--generations 10 --population 20 donor_random False",
    "--generations 10 --population 50 donor_random True",
    "--generations 10 --population 50 donor_random False",
    "--generations 10 --population 100 donor_random True",
    "--generations 10 --population 100 donor_random False",
]

# Iterate over each configuration and execute the script sequentially
for config in configs:
    # Create a unique log file name based on the parameters
    log_file = "run_" + config.replace(" ", "_").replace("=", "_") + ".out"
    log_file = re.sub(r'[^\w\-]', '_', log_file)
    
    # Build the full command
    command = f"python3 diff_evo.py {config}"
    
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