#!/usr/bin/env python3
"""
Script para criar folds estratificados a partir de um CSV e gerar arquivos no formato SVM-Rank.

Este script:
1. Carrega um CSV de entrada com colunas 'query', 'document', 'relevance' e features
2. Divide os dados em K folds estratificados (garantindo que queries não se repetem entre train/test)
3. Gera arquivos train.txt e test.txt no formato SVM-Rank para cada fold
4. Salva os folds em diretórios separados (Fold1/, Fold2/, etc.)

Formato SVM-Rank:
<relevance> qid:<query_id> <feature_id>:<feature_value> ... # <document_id>

Uso:
python scripts/create_stratified_folds.py --input data/colecao_manual_bq.csv --output /path/to/output --n_folds 5
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from typing import List, Tuple, Dict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_feature_map(feature_map_path: str) -> List[str]:
    """Carrega o mapeamento de features do arquivo JSON."""
    try:
        with open(feature_map_path, 'r') as f:
            feature_map = json.load(f)
        logger.info(f"Carregadas {len(feature_map)} features do mapeamento")
        return feature_map
    except Exception as e:
        logger.error(f"Erro ao carregar feature_map.json: {e}")
        raise


def load_and_validate_csv(csv_path: str, feature_map: List[str]) -> pd.DataFrame:
    """Carrega e valida o CSV de entrada."""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Carregado CSV com {len(df)} linhas e {len(df.columns)} colunas")
        
        # Verificar colunas obrigatórias
        required_cols = ['query', 'document', 'relevance'] + feature_map
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Colunas obrigatórias não encontradas: {missing_cols}")
            raise ValueError(f"Colunas obrigatórias não encontradas: {missing_cols}")
        
        # Preencher valores NaN com 0 para as features
        for feature in feature_map:
            df[feature] = df[feature].fillna(0)
        
        logger.info(f"Dataset validado: {df['query'].nunique()} queries únicas, {df['document'].nunique()} documentos únicos")
        return df
        
    except Exception as e:
        logger.error(f"Erro ao carregar CSV: {e}")
        raise


def create_stratified_folds(df: pd.DataFrame, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Cria folds estratificados garantindo que queries não se repetem entre train/test.
    
    Usa StratifiedGroupKFold onde:
    - groups = queries (para garantir que uma query não apareça em train e test)
    - stratification = distribuição de relevância por query
    """
    # Criar um target para estratificação baseado na relevância média por query
    query_relevance = df.groupby('query')['relevance'].agg(['mean', 'count']).reset_index()
    
    # Criar bins de relevância para estratificação
    query_relevance['relevance_bin'] = pd.cut(
        query_relevance['mean'], 
        bins=5, 
        labels=['low', 'low_med', 'medium', 'med_high', 'high']
    )
    
    # Mapear de volta para o dataframe original
    query_to_bin = dict(zip(query_relevance['query'], query_relevance['relevance_bin']))
    df['query_relevance_bin'] = df['query'].map(query_to_bin)
    
    # Preparar dados para o StratifiedGroupKFold
    unique_queries = df['query'].unique()
    query_bins = [query_to_bin[q] for q in unique_queries]
    
    # Criar os folds
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    folds = []
    query_indices = np.arange(len(unique_queries))
    
    for train_query_idx, test_query_idx in sgkf.split(query_indices, query_bins, groups=query_indices):
        train_queries = unique_queries[train_query_idx]
        test_queries = unique_queries[test_query_idx]
        
        train_mask = df['query'].isin(train_queries)
        test_mask = df['query'].isin(test_queries)
        
        train_indices = df[train_mask].index.values
        test_indices = df[test_mask].index.values
        
        folds.append((train_indices, test_indices))
        
        logger.info(f"Fold criado: {len(train_queries)} queries de treino, {len(test_queries)} queries de teste")
        logger.info(f"  Amostras: {len(train_indices)} treino, {len(test_indices)} teste")
    
    return folds


def convert_to_svm_format(df_subset: pd.DataFrame, feature_map: List[str]) -> List[str]:
    """
    Converte um subset do DataFrame para o formato SVM-Rank.
    
    Formato: <relevance> qid:<query_id> <feature_id>:<feature_value> ... # <document_id>
    """
    # Criar mapeamento de queries para IDs numéricos
    unique_queries = df_subset['query'].unique()
    query_to_id = {query: idx + 1 for idx, query in enumerate(unique_queries)}
    
    svm_lines = []
    
    for _, row in df_subset.iterrows():
        relevance = int(row['relevance'])
        query_id = query_to_id[row['query']]
        document_id = row['document']
        
        # Construir features no formato SVM
        feature_pairs = []
        for idx, feature in enumerate(feature_map, 1):
            value = row[feature]
            if pd.notna(value) and value != 0:  # Incluir features com valor 0 também
                feature_pairs.append(f"{idx}:{value}")
            else:
                feature_pairs.append(f"{idx}:0")  # Garantir que todas as features estão presentes
        
        features_str = " ".join(feature_pairs)
        svm_line = f"{relevance} qid:{query_id} {features_str} # {document_id}"
        svm_lines.append(svm_line)
    
    return svm_lines


def save_fold_files(df: pd.DataFrame, fold_indices: Tuple[np.ndarray, np.ndarray], 
                   fold_num: int, output_dir: Path, feature_map: List[str]) -> None:
    """Salva os arquivos de treino e teste para um fold específico."""
    train_indices, test_indices = fold_indices
    
    # Criar diretório do fold
    fold_dir = output_dir / f"Fold{fold_num}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Preparar dados de treino e teste
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()
    
    # Converter para formato SVM
    train_svm = convert_to_svm_format(train_df, feature_map)
    test_svm = convert_to_svm_format(test_df, feature_map)
    
    # Salvar arquivos
    train_file = fold_dir / "train.txt"
    test_file = fold_dir / "test.txt"
    
    with open(train_file, 'w') as f:
        f.write("\n".join(train_svm))
    
    with open(test_file, 'w') as f:
        f.write("\n".join(test_svm))
    
    logger.info(f"Fold{fold_num} salvo: {len(train_svm)} amostras de treino, {len(test_svm)} amostras de teste")
    logger.info(f"  Arquivos: {train_file}, {test_file}")


def main():
    parser = argparse.ArgumentParser(description="Criar folds estratificados para LambdaMART")
    parser.add_argument("--input", "-i", required=True, help="Caminho para o arquivo CSV de entrada")
    parser.add_argument("--output", "-o", required=True, help="Diretório de saída para os folds")
    parser.add_argument("--n_folds", "-k", type=int, default=5, help="Número de folds (padrão: 5)")
    parser.add_argument("--feature_map", "-f", default="data/feature_map.json", 
                       help="Caminho para o feature_map.json (padrão: data/feature_map.json)")
    
    args = parser.parse_args()
    
    # Verificar se os arquivos de entrada existem
    if not os.path.exists(args.input):
        logger.error(f"Arquivo CSV não encontrado: {args.input}")
        return 1
    
    if not os.path.exists(args.feature_map):
        logger.error(f"Feature map não encontrado: {args.feature_map}")
        return 1
    
    # Criar diretório de saída
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Carregar mapeamento de features
        feature_map = load_feature_map(args.feature_map)
        
        # Carregar e validar dados
        df = load_and_validate_csv(args.input, feature_map)
        
        # Criar folds estratificados
        logger.info(f"Criando {args.n_folds} folds estratificados...")
        folds = create_stratified_folds(df, args.n_folds)
        
        # Salvar cada fold
        for i, fold_indices in enumerate(folds, 1):
            save_fold_files(df, fold_indices, i, output_dir, feature_map)
        
        logger.info(f"Processamento concluído! {args.n_folds} folds salvos em {output_dir}")
        
        # Estatísticas finais
        logger.info("\n=== Estatísticas Finais ===")
        logger.info(f"Total de queries: {df['query'].nunique()}")
        logger.info(f"Total de documentos: {df['document'].nunique()}")
        logger.info(f"Total de amostras: {len(df)}")
        logger.info(f"Features utilizadas: {len(feature_map)}")
        logger.info(f"Folds criados: {args.n_folds}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Erro durante o processamento: {e}")
        return 1


if __name__ == "__main__":
    exit(main())