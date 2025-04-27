import yfinance as yf
import os
from datetime import datetime

# Ticker da ação que queremos baixar os dados
TICKER = "PETR4.SA"
# Período histórico para baixar os dados
# Começamos com um período razoável, podemos ajustar depois
DATA_INICIO = "2023-01-01"
DATA_FIM = datetime.now().strftime("%Y-%m-%d")

# Diretório onde os dados brutos serão salvos
DIRETORIO_RAW_DATA = "data/raw/"

# Nome do arquivo de saída
NOME_ARQUIVO = f"{TICKER}_data.csv"
CAMINHO_ARQUIVO_SAIDA = os.path.join(DIRETORIO_RAW_DATA, NOME_ARQUIVO)


# --- Função para baixar e salvar dados ---
def baixar_e_salvar_dados(
    ticker: str, data_inicio: str, data_fim: str, caminho_saida: str
):
    """
    Baixa dados históricos de uma ação usando yfinance e salva em CSV.

    Args:
        ticker (str): O ticker da ação (ex: 'AAPL').
        data_inicio (str): Data de início para baixar (formato 'YYYY-MM-DD').
        data_fim (str): Data de fim para baixar (formato 'YYYY-MM-DD').
        caminho_saida (str): Caminho completo para salvar o arquivo CSV.
    """
    print(f"Baixando dados para {ticker} de {data_inicio} até {data_fim}...")

    try:
        # Usa yfinance para baixar os dados
        dados = yf.download(ticker, start=data_inicio, end=data_fim)

        # Verifica se os dados foram baixados corretamente
        if dados.empty:
            print(
                f"Aviso: Nenhum dado encontrado para o ticker {ticker} no período especificado."
            )
            return

        # Cria o diretório de saída se ele não existir
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

        # Salva os dados em um arquivo CSV
        dados.to_csv(caminho_saida)

        print(f"Dados salvos com sucesso em {caminho_saida}")

    except Exception as e:
        print(f"Erro ao baixar ou salvar dados para {ticker}: {e}")


# --- Execução ---
if __name__ == "__main__":
    # Chama a função principal quando o script é executado diretamente
    baixar_e_salvar_dados(TICKER, DATA_INICIO, DATA_FIM, CAMINHO_ARQUIVO_SAIDA)
