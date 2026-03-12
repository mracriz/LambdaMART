# Scripts para Criação de Folds Estratificados

Este diretório contém scripts para dividir dados de ranking em folds estratificados, garantindo que queries não se repetem entre treino e teste.

## Scripts Disponíveis

### 1. `create_stratified_folds.py` (Principal)

Script principal que realiza a divisão estratificada dos dados.

**Características:**
- Divisão estratificada baseada na distribuição de relevância por query
- Garante que queries não aparecem tanto no treino quanto no teste
- Gera arquivos no formato SVM-Rank
- Inclui todas as features mesmo que sejam zero
- Logging detalhado do processo

**Uso:**
```bash
python3 scripts/create_stratified_folds.py \
    --input data/colecao_manual_bq.csv \
    --output /caminho/para/saida \
    --n_folds 5 \
    --feature_map data/feature_map.json
```

**Parâmetros:**
- `--input, -i`: Arquivo CSV de entrada (obrigatório)
- `--output, -o`: Diretório de saída (obrigatório)
- `--n_folds, -k`: Número de folds (padrão: 5)
- `--feature_map, -f`: Arquivo de mapeamento de features (padrão: data/feature_map.json)

### 2. `create_folds_interactive.py` (Interativo)

Script com interface interativa mais amigável para usuários.

**Uso:**
```bash
python3 scripts/create_folds_interactive.py
```

O script irá solicitar:
1. Seleção do arquivo CSV (da pasta data/ ou caminho personalizado)
2. Diretório de saída
3. Número de folds
4. Confirmação antes de executar

## Formato de Entrada

O CSV deve conter as seguintes colunas obrigatórias:
- `query`: Identificador da query
- `document`: Identificador do documento
- `relevance`: Label de relevância (numérico)
- Features definidas em `feature_map.json`

## Formato de Saída

### Estrutura de Diretórios
```
output_directory/
├── Fold1/
│   ├── train.txt
│   └── test.txt
├── Fold2/
│   ├── train.txt
│   └── test.txt
└── ...
```

### Formato SVM-Rank
Cada linha nos arquivos segue o formato:
```
<relevance> qid:<query_id> <feature_id>:<feature_value> ... # <document_id>
```

Exemplo:
```
2 qid:1 1:101.802025 2:3.662675 3:0 4:0 5:80.33733 ... # JURISPRUDENCE:2873279743
```

## Estratificação

A estratificação é realizada usando `StratifiedGroupKFold` onde:
- **Groups**: Queries (garantem que uma query não aparece em treino e teste)
- **Stratification**: Distribuição de relevância média por query
- **Bins**: 5 níveis de relevância (low, low_med, medium, med_high, high)

## Dependências

```bash
pip install pandas scikit-learn numpy
```

## Exemplos

### Exemplo Básico
```bash
# Criar 5 folds com dados padrão
python3 scripts/create_stratified_folds.py \
    -i data/colecao_manual_bq.csv \
    -o /Users/david/Documents/data/goldenset_paper/svm-format/manual
```

### Exemplo com 10 Folds
```bash
# Criar 10 folds
python3 scripts/create_stratified_folds.py \
    -i data/colecao_manual_bq.csv \
    -o /tmp/my_folds \
    -k 10
```

### Exemplo Interativo
```bash
# Modo interativo (mais fácil)
python3 scripts/create_folds_interactive.py
```

## Validação

Após a execução, o script reporta:
- Número de queries únicas por fold
- Número de amostras por fold
- Distribuição entre treino e teste
- Estatísticas finais

## Notas Importantes

1. **Queries Únicas**: Uma query nunca aparece tanto no treino quanto no teste do mesmo fold
2. **Features Zero**: Todas as features são incluídas, mesmo com valor 0
3. **Ordem de Features**: Segue a ordem definida em `feature_map.json`
4. **Query IDs**: São gerados automaticamente para cada fold (não são globais)
5. **Reprodutibilidade**: Usa `random_state=42` para resultados consistentes