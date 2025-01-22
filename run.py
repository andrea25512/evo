import subprocess
import re

# Define parameter combinations
configs1 = [
    "-a -g 10 -p 10 -c 10 -s 0 -r 0",
    "-a -g 10 -p 10 -c 10 -s 0 -r 1",
    "-a -g 10 -p 10 -c 10 -s 0 -r 2",
    "-a -g 10 -p 10 -c 10 -s 1 -r 0",
    "-a -g 10 -p 10 -c 10 -s 1 -r 1",
    "-a -g 10 -p 10 -c 10 -s 1 -r 2",
    "-a -g 10 -p 10 -c 10 -s 2 -r 0",
    "-a -g 10 -p 10 -c 10 -s 2 -r 1",
    "-a -g 10 -p 10 -c 10 -s 2 -r 2",
    "-a -g 10 -p 10 -c 20 -s 0 -r 0",
    "-a -g 10 -p 10 -c 20 -s 0 -r 1",
    "-a -g 10 -p 10 -c 20 -s 0 -r 2",
    "-a -g 10 -p 10 -c 20 -s 1 -r 0",
    "-a -g 10 -p 10 -c 20 -s 1 -r 1",
    "-a -g 10 -p 10 -c 20 -s 1 -r 2",
    "-a -g 10 -p 10 -c 20 -s 2 -r 0",
    "-a -g 10 -p 10 -c 20 -s 2 -r 1",
    "-a -g 10 -p 10 -c 20 -s 2 -r 2",
    "-a -g 10 -p 50 -c 50 -s 0 -r 0",
    "-a -g 10 -p 50 -c 50 -s 0 -r 1",
    "-a -g 10 -p 50 -c 50 -s 0 -r 2",
    "-a -g 10 -p 50 -c 50 -s 1 -r 0",
    "-a -g 10 -p 50 -c 50 -s 1 -r 1",
    "-a -g 10 -p 50 -c 50 -s 1 -r 2",
    "-a -g 10 -p 50 -c 50 -s 2 -r 0",
    "-a -g 10 -p 50 -c 50 -s 2 -r 1",
    "-a -g 10 -p 50 -c 50 -s 2 -r 2",
    "-a -g 10 -p 50 -c 100 -s 0 -r 0",
    "-a -g 10 -p 50 -c 100 -s 0 -r 1",
    "-a -g 10 -p 50 -c 100 -s 0 -r 2",
    "-a -g 10 -p 50 -c 100 -s 1 -r 0",
    "-a -g 10 -p 50 -c 100 -s 1 -r 1",
    "-a -g 10 -p 50 -c 100 -s 1 -r 2",
    "-a -g 10 -p 50 -c 100 -s 2 -r 0",
    "-a -g 10 -p 50 -c 100 -s 2 -r 1",
    "-a -g 10 -p 50 -c 100 -s 2 -r 2"
]

# Iterate over each configuration and execute the script sequentially
for config in configs1:
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

# Define parameter combinations
configs2 = [
    "-g 10 -p 10 -f",
    "-g 10 -p 10 -f -d",
    "-g 10 -p 20 -f",
    "-g 10 -p 20 -f -d",
    "-g 10 -p 50 -f",
    "-g 10 -p 50 -f -d",
    "-g 10 -p 100 -f",
    "-g 10 -p 100 -f -d"

    "-g 10 -p 10 -a",
    "-g 10 -p 10 -a -d",
    "-g 10 -p 20 -a",
    "-g 10 -p 20 -a -d",
    "-g 10 -p 50 -a",
    "-g 10 -p 50 -a -d",
    "-g 10 -p 100 -a",
    "-g 10 -p 100 -a -d"
]


# Iterate over each configuration and execute the script sequentially
for config in configs2:
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