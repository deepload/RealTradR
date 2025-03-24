# Docker Setup for RealTradR

This document explains how to use Docker to run RealTradR in a containerized environment, which helps avoid TensorFlow DLL loading issues and ensures consistent behavior across different environments.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

## Files Created

1. **Dockerfile**: Defines the container environment with Python 3.10 and all necessary dependencies
2. **docker-compose.yml**: Orchestrates the RealTradR application and its PostgreSQL database
3. **run_docker.ps1**: PowerShell script to build and run the Docker container (Windows)
4. **run_docker.sh**: Bash script to build and run the Docker container (Linux/Mac)

## Running RealTradR in Docker

### Option 1: Using the provided scripts

**On Windows:**
```powershell
.\run_docker.ps1
```

**On Linux/Mac:**
```bash
chmod +x run_docker.sh
./run_docker.sh
```

### Option 2: Manual Docker commands

```bash
# Build the Docker image
docker-compose build

# Start the containers
docker-compose up

# To run in detached mode (background)
docker-compose up -d

# To stop the containers
docker-compose down
```

## Testing in Docker

To run tests inside the Docker container:

```bash
# Run a specific test
docker-compose run --rm realtradR python -m pytest tests/test_production_readiness.py -v

# Run all tests
docker-compose run --rm realtradR python -m pytest
```

## Benefits of Using Docker

1. **Consistent Environment**: The same environment is used for development, testing, and production
2. **Dependency Management**: All dependencies, including TensorFlow, are properly installed and configured
3. **Isolation**: The application runs in isolation, avoiding conflicts with other software
4. **Portability**: The containerized application can run on any system with Docker installed
5. **Scalability**: Easy to scale up by deploying multiple containers

## Container Structure

- The application code is mounted as a volume, so changes to the code are immediately reflected
- The database data is persisted in a Docker volume
- Logs and model files are stored in mounted volumes for persistence

## Environment Variables

Environment variables are loaded from the `.env` file. Make sure your Alpaca API credentials are correctly set in this file.

## Troubleshooting

If you encounter any issues:

1. Check the Docker logs:
   ```bash
   docker-compose logs
   ```

2. Access the container shell:
   ```bash
   docker-compose exec realtradR bash
   ```

3. Verify TensorFlow is working inside the container:
   ```bash
   docker-compose exec realtradR python -c "import tensorflow as tf; print(tf.__version__)"
   ```
