#!/bin/bash

# Define distribution name
DIST_NAME="valsci-docker-dist"
DIST_DIR="./${DIST_NAME}"

echo "Creating Valsci Docker distribution package..."

# Create distribution directory
mkdir -p $DIST_DIR

# Build Docker image if it doesn't exist
if [[ "$(docker images -q valsci:latest 2> /dev/null)" == "" ]]; then
  echo "Building Docker image..."
  docker build -t valsci:latest .
fi

# Check if the image size is reasonable (less than 1GB)
IMAGE_SIZE=$(docker images valsci:latest --format "{{.Size}}")
echo "Docker image size: $IMAGE_SIZE"

# Check if image is suspiciously large
if [[ "$IMAGE_SIZE" == *GB* ]]; then
  SIZE_NUM=$(echo $IMAGE_SIZE | sed 's/GB//')
  if (( $(echo "$SIZE_NUM > 1.0" | bc -l) )); then
    echo "WARNING: The Docker image is larger than 1GB. This might indicate that data directories were included."
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Aborting. Please check your .dockerignore file to ensure large directories are excluded."
      exit 1
    fi
  fi
fi

# Save Docker image
echo "Saving Docker image (this may take several minutes)..."
docker save valsci:latest | gzip > $DIST_DIR/valsci-image.tar.gz

# Copy necessary files
echo "Copying configuration files..."
cp docker-compose.yml $DIST_DIR/
cp env_vars.json.example $DIST_DIR/
cp DOCKER_README.md $DIST_DIR/
cp DISTRIBUTION_README.md $DIST_DIR/README.md

# Create tarball
echo "Creating distribution tarball..."
tar -czvf "${DIST_NAME}.tar.gz" $DIST_DIR

# Cleanup
echo "Cleaning up temporary files..."
rm -rf $DIST_DIR

echo "Distribution package created: ${DIST_NAME}.tar.gz"
echo "Share this file with your collaborator along with the instructions."
echo "File size: $(du -h ${DIST_NAME}.tar.gz | cut -f1)" 