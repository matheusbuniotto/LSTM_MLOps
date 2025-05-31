FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir -r requirements.txt


COPY . .

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose ports for MLflow and API
EXPOSE 8081 8000

# Run entry 
CMD ["/app/start.sh"]

