import subprocess
import re

# Define parameter combinations
configs1 = [
    "-f -g 10 -p 10 -c 10 -s 0 -r 1",
    "-f -g 10 -p 10 -c 10 -s 0 -r 2",
    "-f -g 10 -p 10 -c 10 -s 1 -r 0",
    "-f -g 10 -p 10 -c 10 -s 1 -r 1",
    "-f -g 10 -p 10 -c 10 -s 1 -r 2",
    "-f -g 10 -p 10 -c 10 -s 2 -r 0",
    "-f -g 10 -p 10 -c 10 -s 2 -r 1",
    "-f -g 10 -p 10 -c 10 -s 2 -r 2",
    "-f -g 10 -p 10 -c 20 -s 0 -r 0",
    "-f -g 10 -p 10 -c 20 -s 0 -r 1",
    "-f -g 10 -p 10 -c 20 -s 0 -r 2",
    "-f -g 10 -p 10 -c 20 -s 1 -r 0",
    "-f -g 10 -p 10 -c 20 -s 1 -r 1",
    "-f -g 10 -p 10 -c 20 -s 1 -r 2",
    "-f -g 10 -p 10 -c 20 -s 2 -r 0",
    "-f -g 10 -p 10 -c 20 -s 2 -r 1",
    "-f -g 10 -p 10 -c 20 -s 2 -r 2",
    "-f -g 10 -p 50 -c 50 -s 0 -r 0",
    "-f -g 10 -p 50 -c 50 -s 0 -r 1",
    "-f -g 10 -p 50 -c 50 -s 0 -r 2",
    "-f -g 10 -p 50 -c 50 -s 1 -r 0",
    "-f -g 10 -p 50 -c 50 -s 1 -r 1",
    "-f -g 10 -p 50 -c 50 -s 1 -r 2",
    "-f -g 10 -p 50 -c 50 -s 2 -r 0",
    "-f -g 10 -p 50 -c 50 -s 2 -r 1",
    "-f -g 10 -p 50 -c 50 -s 2 -r 2",
    "-f -g 10 -p 50 -c 100 -s 0 -r 0",
    "-f -g 10 -p 50 -c 100 -s 0 -r 1",
    "-f -g 10 -p 50 -c 100 -s 0 -r 2",
    "-f -g 10 -p 50 -c 100 -s 1 -r 0",
    "-f -g 10 -p 50 -c 100 -s 1 -r 1",
    "-f -g 10 -p 50 -c 100 -s 1 -r 2",
    "-f -g 10 -p 50 -c 100 -s 2 -r 0",
    "-f -g 10 -p 50 -c 100 -s 2 -r 1",
    "-f -g 10 -p 50 -c 100 -s 2 -r 2"
]

configs11 = [
    "-f -g 10 -p 50 -c 50 -s 0 -r 0"
]

configs2 = [
    "-g 10 -p 10 -d",
    "-g 10 -p 10",
    "-g 10 -p 20 -d",
    "-g 10 -p 20",
    "-g 10 -p 50 -d",
    "-g 10 -p 50",
    "-g 10 -p 100 -d",
    "-g 10 -p 100"
]

# Iterate over each configuration and execute the script sequentially
for config in configs11:
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