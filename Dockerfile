FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt requirements.txt

# Install ONLY your *other* dependencies (MLflow, pandas, etc.)
# You might need to install system dependencies here if required by other packages
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils \
  # Add any other system libraries required by your dependencies here
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir -r requirements.txt

# Copy the start script and make it executable
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Copy the rest of your application code
COPY . .

# Expose the port MLflow will run on
EXPOSE 8081

# Use the start script as the entrypoint
CMD ["/app/start.sh"]

