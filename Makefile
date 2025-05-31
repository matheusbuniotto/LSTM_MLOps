# Makefile para o projeto LSTM_MLOps

# Variáveis
IMAGE_NAME := lstm_mlflow
CONTAINER_NAME := mlflow-server
PORT := 8081
PROJECT_DIR := $(shell pwd)
UV := uv

# Targets
.PHONY: help
help:
	@echo "Makefile para o projeto LSTM_MLOps"
	@echo "Targets:"
	@echo "  help           - Mostra esta mensagem de ajuda"
	@echo "  build          - Constrói a imagem Docker"
	@echo "  run            - Executa o contêiner Docker com o servidor MLflow"
	@echo "  stop           - Para e remove o contêiner Docker"
	@echo "  logs           - Visualiza os logs do contêiner Docker"
	@echo "  test           - Executa scripts/test.py (localmente com uv)"
	@echo "  process-data   - Executa data/process_data.py (localmente com uv)"
	@echo "  test-docker    - Executa scripts/test.py dentro do contêiner Docker"
	@echo "  install        - Instala dependências localmente"
	@echo "  clean          - Limpa imagens e contêineres Docker"
	@echo "  predict        - Executa scripts/predict.py localmente"
	@echo "  predict-docker - Executa scripts/predict.py dentro do contêiner Docker"
	@echo "  predict-api-docker - Executa a API de predição dentro do contêiner Docker"

# Build na imagem Docker
.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .

# Executa o Docker
.PHONY: run
run:
	@if [ "$(shell docker ps -aq -f name=$(CONTAINER_NAME))" ]; then \
		if [ "$(shell docker ps -q -f status=running -f name=$(CONTAINER_NAME))" ]; then \
			echo "Container $(CONTAINER_NAME) já está em execução."; \
		else \
			echo "Reiniciando o container parado $(CONTAINER_NAME)..."; \
			docker start $(CONTAINER_NAME); \
		fi \
	else \
		docker run -d --name $(CONTAINER_NAME) -p 8081:8081 -p 8000:8000 $(IMAGE_NAME); \
	fi
	@echo "\nMLflow UI disponível em: http://localhost:$(PORT)"
	@echo "API disponível em: http://localhost:8000"
	@echo "Outros links úteis:"
	@echo "- Documentação MLflow: https://mlflow.org/docs/latest/index.html"

# Para e remove o contêiner Docker
.PHONY: stop
stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# Visualiza os logs do contêiner Docker
.PHONY: logs
logs:
	docker logs $(CONTAINER_NAME)

# Executa test.py localmente com uv
.PHONY: test
test:
	$(UV) run scripts/test.py

# Executa process_data.py localmente com uv
.PHONY: process-data
process-data:
	$(UV) run data/process_data.py

# Executa test.py dentro do contêiner Docker
.PHONY: test-docker
test-docker:
	@echo "\n🔍 Verificando se o contêiner $(CONTAINER_NAME) está em execução..."
	@if [ ! "$$(docker ps -q -f name=$(CONTAINER_NAME))" ]; then \
		echo "❌ Contêiner $(CONTAINER_NAME) não está em execução. Executando..."; \
		$(MAKE) run; \
		echo "⏳ Aguardando inicialização do MLflow (5s)..."; \
		sleep 5; \
	else \
		echo "✅ Contêiner $(CONTAINER_NAME) já está em execução."; \
	fi
	@echo "\n🧪 Executando testes no contêiner..."
	@docker exec -e MLFLOW_TRACKING_URI=http://localhost:$(PORT) $(CONTAINER_NAME) python /app/scripts/test.py; \
	EXIT_CODE=$$?; \
	if [ $$EXIT_CODE -eq 0 ]; then \
		echo "\n✅ TESTE DOCKER: OK - Os testes foram concluídos com sucesso!"; \
	else \
		echo "\n❌ TESTE DOCKER: FALHA - Os testes falharam com código de saída $$EXIT_CODE"; \
	fi; \
	exit $$EXIT_CODE

# Instala dependências localmente
.PHONY: install
install:
	$(UV) pip install -r requirements.txt

# Limpa recursos Docker
.PHONY: clean
clean:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
	docker rmi $(IMAGE_NAME) || true
	echo "Removendo TODOS os containers Docker..."
	docker rm -f $(docker ps -aq) || true
	echo "Removendo TODAS as imagens Docker..."
	docker rmi -f $(docker images -aq) || true

# Executa scripts/predict.py
.PHONY: predict
predict:
	@echo "Running prediction script locally..."
	@python scripts/predict.py

# Executa scripts/predict.py dentro do contêiner Docker
.PHONY: predict-docker
predict-docker:
	@echo "\n🔍 Verificando se o contêiner $(CONTAINER_NAME) está em execução..."
	@if [ ! "$$(docker ps -q -f name=$(CONTAINER_NAME))" ]; then \
		echo "❌ Contêiner $(CONTAINER_NAME) não está em execução. Executando..."; \
		$(MAKE) run; \
		echo "⏳ Aguardando inicialização do MLflow (5s)..."; \
		sleep 5; \
	else \
		echo "✅ Contêiner $(CONTAINER_NAME) já está em execução."; \
	fi
	@echo "\n🤖 Executando predição no contêiner..."
	@docker exec -e MLFLOW_TRACKING_URI=http://localhost:$(PORT) $(CONTAINER_NAME) python /app/scripts/predict.py; \
	EXIT_CODE=$$?; \
	if [ $$EXIT_CODE -eq 0 ]; then \
		echo "\n✅ PREDIÇÃO DOCKER: OK - A predição foi concluída com sucesso!"; \
	else \
		echo "\n❌ PREDIÇÃO DOCKER: FALHA - A predição falhou com código de saída $$EXIT_CODE"; \
	fi; \
	exit $$EXIT_CODE

# Executa a API de predição dentro do contêiner Docker
.PHONY: predict-api-docker
predict-api-docker:
	@echo "\n🔍 Verificando se o contêiner $(CONTAINER_NAME) está em execução..."
	@if [ ! "$$(docker ps -q -f name=$(CONTAINER_NAME))" ]; then \
		echo "❌ Contêiner $(CONTAINER_NAME) não está em execução. Executando..."; \
		$(MAKE) run; \
		echo "⏳ Aguardando inicialização do container (5s)..."; \
		sleep 5; \
	else \
		echo "✅ Contêiner $(CONTAINER_NAME) já está em execução."; \
	fi
	@echo "\n🤖 Executando API de predição no contêiner..."
	@docker exec $(CONTAINER_NAME) uvicorn api:app --host 0.0.0.0 --port 8000
