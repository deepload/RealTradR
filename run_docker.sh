#!/bin/bash

# Bash script to build and run RealTradR in Docker

# Stop any existing containers
echo "Stopping any existing RealTradR containers..."
docker-compose down

# Build the Docker image
echo "Building the RealTradR Docker image..."
docker-compose build

# Run the container
echo "Starting RealTradR in Docker..."
docker-compose up

# Note: To run in detached mode, use:
# docker-compose up -d
