# LambdaMART Training Pipeline

Um pipeline completo e modular para treinamento de modelos LambdaMART com suporte a múltiplos frameworks, IPS (Inverse Propensity Scoring) para debiasing, e hyperparameter tuning automático.

## 🚀 Características Principais

### Frameworks Suportados
- **XGBoost**: Implementação robusta com suporte completo ao IPS
- **LightGBM**: Implementação nativa otimizada para velocidade

### Funcionalidades Avançadas
- ✅ **IPS (Inverse Propensity Scoring)**: Remove viés de posição nos dados
- ✅ **Hyperparameter Tuning**: Otimização automática de hiperparâmetros
- ✅ **MLflow Integration**: Rastreamento completo de experimentos
- ✅ **Configurações Flexíveis**: Sistema baseado em YAML para diferentes cenários
- ✅ **Avaliação Completa**: NDCG@k e MRR com múltiplos valores de k

### Métricas de Avaliação
- **NDCG@1, @3, @5, @10**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **Ranking Quality**: Análise completa da qualidade do ranking

## 📁 Estrutura do Projeto

```
LambdaMART/
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Carregamento de dados SVM-Rank
│   ├── model.py             # Implementação XGBoost + IPS
│   ├── model_lightgbm.py    # Implementação LightGBM nativa
│   ├── evaluator.py         # Métricas de avaliação
│   └── mlflow_utils.py      # Integração com MLflow
├── configs/
│   ├── config.yaml                    # Configuração básica XGBoost
│   ├── config_ips.yaml               # XGBoost + IPS para debiasing
│   ├── config_lightgbm.yaml          # LightGBM baseline
│   ├── config_lightgbm_tuning.yaml   # LightGBM com hyperparameter tuning
│   └── config_lightgbm_test_tuning.yaml  # Teste rápido de tuning
├── main.py                  # Script principal unificado
├── pyproject.toml           # Dependências gerenciadas pelo Poetry
├── poetry.lock              # Lockfile do Poetry
└── README.md               # Este arquivo
```

## 🛠️ Instalação

### Pré-requisitos
- Python 3.9+
- [Poetry](https://python-poetry.org/docs/#installation)
- `libomp` (macOS, necessário para XGBoost): `brew install libomp`

### Instalação
```bash
git clone <repository-url>
cd LambdaMART
poetry install
```

Para ativar o ambiente virtual:
```bash
poetry shell
```

Ou para executar comandos dentro do ambiente sem ativá-lo:
```bash
poetry run python main.py --config configs/config.yaml
```

## 📖 Uso

### 1. Treinamento Básico (XGBoost)
```bash
python main.py --config configs/config.yaml
```

### 2. Treinamento com IPS (Debiasing)
```bash
python main.py --config configs/config_ips.yaml
```

### 3. Treinamento com LightGBM
```bash
python main.py --config configs/config_lightgbm.yaml
```

### 4. Hyperparameter Tuning
```bash
# Teste rápido (4 combinações)
python main.py --config configs/config_lightgbm_test_tuning.yaml

# Tuning completo (64 combinações)
python main.py --config configs/config_lightgbm_tuning.yaml
```

### 5. Automação de Múltiplos Experimentos
```bash
# Teste rápido de todas as combinações (sem tuning)
python3 run_quick_test.py

# Execução completa com hyperparameter tuning
python3 run_all_experiments.py
```

### 6. Especificar Framework Manualmente
```bash
python main.py --config configs/config.yaml --framework lightgbm
```

### 7. Execução Manual com Variáveis de Ambiente
```bash
# Executar experimento específico
LAMBDAMART_SIGNAL=click LAMBDAMART_METHOD=CCM python3 main.py --config configs/config_lightgbm.yaml
```

## ⚙️ Configuração

### Arquivo de Configuração Base
```yaml
# Caminhos dos dados
data:
  train_dir: "/path/to/train/data"     # Arquivos .txt de treino
  test_dir: "/path/to/test/data"       # Arquivo .txt de teste

# Configurações MLflow
mlflow:
  experiment_name: "LambdaMART_Experiment"
  log_model: true
  log_predictions: true

# Parâmetros do modelo
model:
  framework: "xgboost"          # ou "lightgbm"
  objective: "rank:ndcg"        # ou "lambdarank" para LightGBM
  learning_rate: 0.1
  max_depth: 6
  # ... outros parâmetros

# Configurações de treinamento
training:
  num_boost_round: 200
  early_stopping_rounds: null   # ou número para early stopping

# Hyperparameter tuning (opcional)
advanced:
  hyperparameter_tuning:
    enabled: false
    search_space:
      learning_rate: [0.05, 0.1, 0.2]
      max_depth: [4, 6, 8]
```

### Configuração IPS (Debiasing)
```yaml
model:
  # Parâmetros IPS específicos
  lambdarank_unbiased: true           # Habilita IPS
  lambdarank_bias_norm: 1.0           # Normalização do viés
  lambdarank_normalization: "query"   # Normalização por query
```

## 📊 Formato dos Dados

### SVM-Rank Format
```
<relevance_score> qid:<query_id> <feature_1>:<value_1> ... <feature_n>:<value_n>
```

### Exemplo
```
3 qid:1 1:0.5 2:0.3 3:0.8
2 qid:1 1:0.4 2:0.7 3:0.2
1 qid:1 1:0.2 2:0.1 3:0.9
3 qid:2 1:0.8 2:0.9 3:0.1
```

### Estrutura de Diretórios
```
data/
├── train/
│   ├── train.txt
│   └── vali.txt      # opcional
└── test/
    └── test.txt
```

## � Automação de Experimentos

### Scripts de Automação Disponíveis

#### 1. Teste Rápido (`run_quick_test.py`)
Valida rapidamente todas as combinações sem hyperparameter tuning:
```bash
python3 run_quick_test.py
```
- **Objetivo**: Verificar se todas as combinações funcionam
- **Tempo**: ~2-3 min por experimento
- **Total**: ~20-30 minutos para todas as combinações

#### 2. Automação Completa (`run_all_experiments.py`)
Executa hyperparameter tuning para todas as combinações:
```bash
python3 run_all_experiments.py
```
- **Objetivo**: Encontrar os melhores hiperparâmetros para cada combinação
- **Tempo**: ~30-60 min por experimento (64 trials cada)
- **Total**: ~5-10 horas para todas as combinações

### Combinações Executadas
O sistema automaticamente executa experimentos para:

**Signals**: `click`, `copy`
**Methods**: `DCM`, `CCM`, `DBN`, `SDBN`, `SDBN2`

**Total**: 10 experimentos (2 × 5 combinações)

### Estrutura dos Dados Esperada
```
/Users/david/Documents/data/goldenset_paper/svm-format/
├── click/
│   ├── DCM/
│   ├── CCM/
│   ├── DBN/
│   ├── SDBN/
│   └── SDBN2/
└── copy/
    ├── DCM/
    ├── CCM/
    ├── DBN/
    ├── SDBN/
    └── SDBN2/
```

### Experimentos MLflow Gerados
Cada combinação gera um experimento separado no MLflow:
- `LambdaMART_LightGBM_DCM_Click`
- `LambdaMART_LightGBM_DCM_Copy`
- `LambdaMART_LightGBM_CCM_Click`
- `LambdaMART_LightGBM_CCM_Copy`
- ... e assim por diante

### Personalização da Automação
Para modificar os hiperparâmetros ou adicionar novas combinações:

1. **Editar hiperparâmetros**: Modifique `configs/config_lightgbm.yaml`
2. **Adicionar signals/methods**: Modifique as listas em `run_all_experiments.py`
3. **Ajustar timeout**: Modifique `timeout=7200` no script (em segundos)

## �🧪 Cenários de Uso

### 1. Comparação de Frameworks
Execute os mesmos dados com diferentes frameworks para comparar performance:
```bash
# XGBoost baseline
python main.py --config configs/config.yaml

# LightGBM baseline  
python main.py --config configs/config_lightgbm.yaml

# XGBoost + IPS (debiasing)
python main.py --config configs/config_ips.yaml
```

### 2. Otimização de Hiperparâmetros
```bash
# Primeiro, teste rápido para validar
python main.py --config configs/config_lightgbm_test_tuning.yaml

# Depois, tuning completo
python main.py --config configs/config_lightgbm_tuning.yaml
```

### 3. Análise de Experimentos
```bash
# Iniciar MLflow UI
mlflow ui

# Acessar: http://localhost:5000
```

## 📈 Resultados Esperados

### Métricas Típicas
- **NDCG@10**: 0.85-0.95 (dependendo do dataset)
- **NDCG@5**: 0.80-0.90
- **MRR**: 0.90-0.98

### Comparação de Frameworks
- **XGBoost**: Mais robusto, melhor com IPS
- **LightGBM**: Mais rápido, boa para baseline
- **XGBoost + IPS**: Melhor para dados com viés de posição

## 🔧 Troubleshooting

### Problemas Comuns

#### 1. Erro de Labels
```
Error: Label X is not less than the number of label mappings
```
**Solução**: Ajustar `label_gain` no config para incluir todos os valores de relevância.

#### 2. Early Stopping sem Validação
```
Warning: Early stopping without validation data
```
**Solução**: Definir `early_stopping_rounds: null` ou adicionar dados de validação.

#### 3. LightGBM não Encontrado
```
ImportError: LightGBM não disponível
```
**Solução**: `pip install lightgbm` ou usar XGBoost apenas.

### Debug e Logs
- Use `verbosity: 1` para logs detalhados
- Verifique MLflow para histórico completo
- Logs salvos automaticamente durante experimentos

## 🤝 Contribuição

### Adicionando Novos Frameworks
1. Criar arquivo `src/model_<framework>.py`
2. Implementar interface compatível
3. Adicionar detecção no `main.py`
4. Criar configurações em `configs/`

### Adicionando Métricas
1. Modificar `src/evaluator.py`
2. Atualizar logs do MLflow
3. Adicionar às configurações

## 📄 Licença

[Especificar licença do projeto]

## 🔗 Referências

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Learning to Rank Literature](https://en.wikipedia.org/wiki/Learning_to_rank)
