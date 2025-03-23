# Use official Rust image
FROM rust:latest

# Install Python and Pip
RUN apt update && apt install -y python3 python3-pip

# Set the working directory inside the container
WORKDIR /workspace
