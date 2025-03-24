# PowerShell script to build and run RealTradR in Docker

# Stop any existing containers
Write-Host "Stopping any existing RealTradR containers..." -ForegroundColor Cyan
docker-compose down

# Build the Docker image
Write-Host "Building the RealTradR Docker image..." -ForegroundColor Cyan
docker-compose build

# Run the container
Write-Host "Starting RealTradR in Docker..." -ForegroundColor Cyan
docker-compose up

# Note: To run in detached mode, use:
# docker-compose up -d
