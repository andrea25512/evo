import subprocess
import re

# Define parameter combinations
configs = [
    "--generations 10 --population 10 --children 10 --selection 0",
    "--generations 10 --population 10 --children 10 --selection 1",
    "--generations 10 --population 10 --children 10 --selection 2",
    "--generations 10 --population 10 --children 20 --selection 0",
    "--generations 10 --population 10 --children 20 --selection 1",
    "--generations 10 --population 10 --children 20 --selection 2",
    "--generations 10 --population 50 --children 100 --selection 0",
    "--generations 10 --population 50 --children 100 --selection 1",
    "--generations 10 --population 50 --children 100 --selection 2"
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