# LambdaMART Model Training Project

Este projeto implementa um pipeline completo para treinamento de modelos LambdaMART usando dados no formato SVM-Rank, com integração ao MLflow para rastreamento de experimentos.

## Características

- **Framework**: XGBoost com algoritmo LambdaMART
- **Formato de Dados**: SVM-Rank (.txt)
- **Métricas**: NDCG@1,3,5,10 e MRR
- **Tracking**: MLflow para experimentos
- **Arquitetura**: Modular e extensível

## Estrutura do Projeto

```
LambdaMART/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Carregamento de dados SVM-Rank
│   ├── model.py            # Implementação do LambdaMART
│   ├── evaluator.py        # Métricas de avaliação
│   └── mlflow_utils.py     # Integração com MLflow
├── data/
│   ├── train/              # Arquivos de treino .txt
│   └── test/               # Arquivos de teste .txt
├── configs/
│   └── config.yaml         # Configurações do modelo
├── notebooks/
│   └── example_usage.ipynb # Exemplo de uso
├── main.py                 # Script principal
└── requirements.txt        # Dependências
```

## Instalação

1. Clone o repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

1. Coloque seus arquivos de treino e teste no formato SVM-Rank nas pastas `data/train/` e `data/test/`
2. Configure os hiperparâmetros em `configs/config.yaml`
3. Execute o treinamento:
```bash
python main.py
```

## Formato dos Dados

O projeto espera arquivos no formato SVM-Rank (.txt):
```
<label> qid:<query_id> <feature_1>:<value_1> ... <feature_n>:<value_n>
```

## Métricas Avaliadas

- NDCG@1, NDCG@3, NDCG@5, NDCG@10
- MRR (Mean Reciprocal Rank)

## MLflow Integration

Todos os experimentos são automaticamente registrados no MLflow, incluindo:
- Hiperparâmetros do modelo
- Métricas de avaliação
- Artefatos do modelo treinado# LambdaMART
