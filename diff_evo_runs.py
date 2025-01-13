import subprocess
import re

# Define parameter combinations
configs = [
    #"--generations 10 --population 10 --donor_random",
    #"--generations 10 --population 10",
    "--generations 10 --population 20 --donor_random",
    "--generations 10 --population 20",
    "--generations 10 --population 50 --donor_random",
    "--generations 10 --population 50",
    "--generations 10 --population 100 --donor_random",
    "--generations 10 --population 100",
]


# Iterate over each configuration and execute the script sequentially
for config in configs:
    # Create a unique log file name based on the parameters
    log_file = "run_" + config.replace(" ", "_").replace("=", "_") + ".out"
    log_file = re.sub(r'[^\w\-]', '_', log_file)
    
    # Build the full command
    command = f"python diff_evo.py {config}"
    
    print(f"Running: {command}")
    
    try:
        with open(log_file, "w") as log:
            subprocess.run(
                command,
                shell=True,  # Allows string-based commands like in the shell
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True  # Raise exception if the command fails
            )
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")