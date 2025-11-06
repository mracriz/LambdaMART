"""
Main script unificado para LambdaMART.
Suporta diferentes configurações: XGBoost padrão, XGBoost com IPS, e comparações.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import SVMRankDataLoader
from model import LambdaMARTModel
from evaluator import RankingEvaluator
from mlflow_utils import MLflowManager
import numpy as np


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Carregar arquivo de configuração YAML.
    
    Args:
        config_path: Caminho para o arquivo de configuração
        
    Returns:
        Dicionário com configurações
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def detect_framework(config: Dict[str, Any]) -> str:
    """
    Detectar qual framework usar baseado na configuração.
    
    Args:
        config: Configuração carregada
        
    Returns:
        Nome do framework ('xgboost' ou 'lightgbm')
    """
    # Verificar se há configuração específica de framework
    if 'framework' in config:
        return config['framework'].lower()
    
    # Detectar baseado nos parâmetros do modelo
    model_params = config.get('model', {})
    
    # Se há parâmetros específicos do XGBoost (especialmente IPS)
    if any(param.startswith('lambdarank_') for param in model_params.keys()):
        return 'xgboost'
    
    # Se há parâmetros específicos do LightGBM
    if 'num_leaves' in model_params or 'boosting_type' in model_params:
        return 'lightgbm'
    
    # Default para XGBoost
    return 'xgboost'


def perform_hyperparameter_tuning(config, train_features, train_labels, train_qids,
                                 test_features, test_labels, test_qids, evaluator, mlflow_manager, framework="xgboost"):
    """
    Realizar tuning de hiperparâmetros.
    """
    import itertools
    
    # Obter espaço de busca da configuração
    search_space = config.get('advanced', {}).get('hyperparameter_tuning', {}).get('search_space', {})
    
    if not search_space:
        print("⚠️ Nenhum espaço de busca definido para hyperparameter tuning")
        return None, None, 0.0
    
    # Gerar todas as combinações
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"🔍 Testando {len(param_combinations)} combinações de parâmetros para {framework.upper()}...")
    
    best_score = 0.0
    best_params = None
    best_model = None
    
    for i, param_combination in enumerate(param_combinations):
        # Criar configuração atual
        current_params = dict(zip(param_names, param_combination))
        trial_params = config['model'].copy()
        trial_params.update(current_params)
        
        print(f"\n🧪 Trial {i+1}/{len(param_combinations)}: {current_params}")
        
        # Iniciar run do MLflow para este trial
        trial_run_name = f"trial_{i+1}_{framework}_{current_params}"
        trial_run_id = mlflow_manager.start_run(
            run_name=trial_run_name,
            tags={"tuning_trial": str(i+1), "source": "hyperparameter_tuning", "framework": framework}
        )
        
        try:
            # Treinar modelo com parâmetros atuais baseado no framework
            if framework == "lightgbm":
                from src.model_lightgbm import LightGBMLambdaMART
                trial_model = LightGBMLambdaMART(trial_params)
            else:
                trial_model = LambdaMARTModel(trial_params)
            
            trial_model.train(
                train_features, train_labels, train_qids,
                num_boost_round=config['training']['num_boost_round'],
                early_stopping_rounds=config['training']['early_stopping_rounds']
            )
            
            # Avaliar
            predictions = trial_model.predict(test_features)
            metrics = evaluator.evaluate_ranking(
                test_labels, predictions, test_qids, 
                k_values=config['evaluation']['k_values']
            )
            
            # Converter valores numpy para float Python (para compatibilidade com MLflow)
            metrics_converted = {}
            for key, value in metrics.items():
                if hasattr(value, 'item'):  # Se for numpy scalar
                    metrics_converted[key] = float(value.item())
                else:
                    metrics_converted[key] = float(value)
            
            # Log no MLflow
            mlflow_manager.log_model_parameters(trial_params)
            mlflow_manager.log_evaluation_metrics(metrics_converted, prefix="test_")
            
            # Verificar se é o melhor
            current_score = metrics_converted.get('ndcg@10', 0.0)
            
            if current_score > best_score:
                best_score = current_score
                best_params = current_params
                best_model = trial_model
                print(f"🎯 Novo melhor score: {current_score:.4f}")
            
        except Exception as e:
            print(f"❌ Erro no trial {i+1}: {e}")
        finally:
            # Finalizar run do trial
            mlflow_manager.end_run()
    
    return best_model, best_params, best_score


def train_and_evaluate(config: Dict[str, Any], framework: str = "xgboost") -> Dict[str, Any]:
    """
    Treinar e avaliar modelo com a configuração especificada.
    
    Args:
        config: Configuração do modelo
        framework: Framework a usar ('xgboost' ou 'lightgbm')
        
    Returns:
        Resultados do experimento
    """
    print(f"🚀 Iniciando experimento com {framework.upper()}...")
    
    # 1. Carregar dados
    print("📁 Carregando dados...")
    data_loader = SVMRankDataLoader()
    data = data_loader.load_train_test_data(
        config['data']['train_dir'], 
        config['data']['test_dir']
    )
    
    train_features = data['train']['features']
    train_labels = data['train']['labels']
    train_qids = data['train']['query_ids']
    
    test_features = data['test']['features']
    test_labels = data['test']['labels']
    test_qids = data['test']['query_ids']
    
    # 2. Inicializar MLflow
    print("📊 Configurando MLflow...")
    mlflow_manager = MLflowManager(
        experiment_name=config['mlflow']['experiment_name'],
        tracking_uri=config['mlflow'].get('tracking_uri')
    )
    
    # 3. Inicializar evaluator
    print("📏 Configurando evaluator...")
    evaluator = RankingEvaluator()
    
    # 4. Inicializar modelo
    print(f"🤖 Inicializando modelo {framework.upper()}...")
    
    if framework == "xgboost":
        model = LambdaMARTModel(config['model'])
        
        # Mostrar informações de IPS se habilitado
        if config['model'].get('lambdarank_unbiased', False):
            print("="*60)
            print("CONFIGURAÇÃO IPS (INVERSE PROPENSITY SCORING)")
            print("="*60)
            print("✅ IPS HABILITADO - Removendo viés de dados de clique")
            print(f"📊 Método de pares: {config['model'].get('lambdarank_pair_method', 'topk')}")
            print(f"🔢 Pares por amostra: {config['model'].get('lambdarank_num_pair_per_sample', 5)}")
            print(f"⚖️ Norma de bias: {config['model'].get('lambdarank_bias_norm', 2.0)}")
            print(f"🎯 Ganho exponencial NDCG: {config['model'].get('ndcg_exp_gain', True)}")
            print("="*60)
    
    elif framework == "lightgbm":
        # Para LightGBM, usar implementação nativa
        try:
            from src.model_lightgbm import LightGBMLambdaMART
            model = LightGBMLambdaMART(config['model'])
        except ImportError:
            print("⚠️ LightGBM não disponível. Usando XGBoost.")
            model = LambdaMARTModel(config['model'])
            framework = "xgboost"
    
    # 5. Verificar se hyperparameter tuning está habilitado
    if config.get('advanced', {}).get('hyperparameter_tuning', {}).get('enabled', False):
        print("🔧 Hyperparameter tuning habilitado!")
        best_model, best_params, best_score = perform_hyperparameter_tuning(
            config, train_features, train_labels, train_qids,
            test_features, test_labels, test_qids, evaluator, mlflow_manager, framework
        )
        model = best_model
        print(f"🏆 Melhor NDCG@10: {best_score:.4f}")
        print(f"🎯 Melhores parâmetros: {best_params}")
    else:
        # 6. Treinar modelo normalmente
        print(f"🏋️ Treinando modelo {framework.upper()}...")
        
        # Iniciar run do MLflow
        run_id = mlflow_manager.start_run(f"{framework}_training")
        
        # Log dos parâmetros no MLflow
        mlflow_manager.log_model_parameters(config['model'])
        
        model.train(
            train_features, train_labels, train_qids,
            num_boost_round=config['training']['num_boost_round'],
            early_stopping_rounds=config['training']['early_stopping_rounds']
        )
    
    # 7. Avaliar
    print(f"📊 Avaliando modelo {framework.upper()}...")
    predictions = model.predict(test_features)
    metrics = evaluator.evaluate_ranking(
        test_labels, predictions, test_qids, 
        k_values=config['evaluation']['k_values']
    )
    
    # Log das métricas no MLflow  
    # Converter valores numpy para float Python
    metrics_converted = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):  # Se for numpy scalar
            metrics_converted[key] = float(value.item())
        else:
            metrics_converted[key] = float(value)
    
    mlflow_manager.log_evaluation_metrics(metrics_converted, prefix="test_")
    
    # Log do modelo como artefato
    if config['mlflow'].get('log_model', True):
        mlflow_manager.log_model_artifact(model.model, f"{framework}_lambdamart_model")
    
    # Log das predições como artefato
    if config['mlflow'].get('log_predictions', True):
        mlflow_manager.log_predictions(predictions, test_labels, test_qids, "test_predictions")
    
    # Finalizar run
    mlflow_manager.end_run()
    
    # 8. Mostrar resultados
    print(f"\n📊 RESULTADOS FINAIS ({framework.upper()}):")
    print("-" * 40)
    for metric_name, value in metrics_converted.items():
        print(f"{metric_name:<15}: {value:.4f}")
    
    return {
        'config': config,
        'framework': framework,
        'model': model,
        'metrics': metrics_converted,
        'predictions': predictions
    }


def convert_xgboost_to_lightgbm_params(xgb_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converter parâmetros XGBoost para LightGBM.
    
    Args:
        xgb_params: Parâmetros XGBoost
        
    Returns:
        Parâmetros LightGBM equivalentes
    """
    lgb_params = {}
    
    # Mapeamento de parâmetros
    param_mapping = {
        'eta': 'learning_rate',
        'max_depth': 'max_depth',
        'subsample': 'bagging_fraction',
        'colsample_bytree': 'feature_fraction',
        'alpha': 'lambda_l1',
        'lambda': 'lambda_l2',
        'min_child_weight': 'min_data_in_leaf',
        'verbosity': 'verbose',
        'seed': 'random_state'
    }
    
    for xgb_key, value in xgb_params.items():
        if xgb_key in param_mapping:
            lgb_key = param_mapping[xgb_key]
            lgb_params[lgb_key] = value
        elif xgb_key.startswith('lambdarank_'):
            # Ignorar parâmetros específicos do XGBoost IPS
            continue
        elif xgb_key in ['objective', 'eval_metric']:
            # Converter objetivos
            if xgb_key == 'objective' and value == 'rank:ndcg':
                lgb_params['objective'] = 'lambdarank'
            elif xgb_key == 'eval_metric' and value == 'ndcg':
                lgb_params['metric'] = 'ndcg'
        else:
            lgb_params[xgb_key] = value
    
    # Parâmetros padrão específicos do LightGBM
    if 'boosting_type' not in lgb_params:
        lgb_params['boosting_type'] = 'gbdt'
    if 'num_leaves' not in lgb_params:
        # Converter max_depth para num_leaves aproximado
        max_depth = lgb_params.get('max_depth', 6)
        lgb_params['num_leaves'] = min(2**max_depth, 255)
    
    return lgb_params


def compare_frameworks(config_paths: List[str]) -> Dict[str, Any]:
    """
    Comparar diferentes configurações/frameworks.
    
    Args:
        config_paths: Lista de caminhos para arquivos de configuração
        
    Returns:
        Resultados da comparação
    """
    print("🔄 Iniciando comparação entre configurações...")
    
    results = {}
    
    for config_path in config_paths:
        config_name = Path(config_path).stem
        print(f"\n{'='*60}")
        print(f"📋 Executando configuração: {config_name}")
        print(f"📁 Arquivo: {config_path}")
        print(f"{'='*60}")
        
        try:
            config = load_config(config_path)
            framework = detect_framework(config)
            result = train_and_evaluate(config, framework)
            results[config_name] = result
        except Exception as e:
            print(f"❌ Erro ao executar {config_name}: {e}")
            results[config_name] = {"error": str(e)}
    
    # Mostrar comparação
    print(f"\n{'='*80}")
    print("📊 COMPARAÇÃO DE RESULTADOS")
    print(f"{'='*80}")
    
    # Cabeçalho
    print(f"{'Configuração':<20} {'Framework':<10} {'NDCG@1':<8} {'NDCG@3':<8} {'NDCG@5':<8} {'NDCG@10':<9} {'MRR':<8}")
    print("-" * 80)
    
    # Resultados
    for config_name, result in results.items():
        if "error" in result:
            print(f"{config_name:<20} {'ERROR':<10} {result['error']}")
        else:
            metrics = result['metrics']
            framework = result['framework'].upper()
            print(f"{config_name:<20} {framework:<10} "
                  f"{metrics.get('ndcg@1', 0):<8.4f} "
                  f"{metrics.get('ndcg@3', 0):<8.4f} "
                  f"{metrics.get('ndcg@5', 0):<8.4f} "
                  f"{metrics.get('ndcg@10', 0):<9.4f} "
                  f"{metrics.get('mrr', 0):<8.4f}")
    
    return results


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="LambdaMART Training Pipeline - Suporte para XGBoost e LightGBM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Executar com configuração padrão
  python main.py --config configs/config.yaml
  
  # Executar com IPS habilitado
  python main.py --config configs/config_ips.yaml
  
  # Comparar múltiplas configurações
  python main.py --compare configs/config.yaml configs/config_ips.yaml
  
  # Forçar framework específico
  python main.py --config configs/config.yaml --framework lightgbm
        """
    )
    
    parser.add_argument('--config', '-c', type=str,
                      help='Caminho para arquivo de configuração YAML')
    parser.add_argument('--compare', nargs='+', type=str,
                      help='Comparar múltiplas configurações')
    parser.add_argument('--framework', '-f', choices=['xgboost', 'lightgbm'],
                      help='Forçar framework específico')
    parser.add_argument('--list-configs', action='store_true',
                      help='Listar configurações disponíveis')
    
    args = parser.parse_args()
    
    # Listar configurações disponíveis
    if args.list_configs:
        config_dir = Path("configs")
        if config_dir.exists():
            print("📋 Configurações disponíveis:")
            for config_file in config_dir.glob("*.yaml"):
                print(f"  - {config_file}")
        else:
            print("❌ Diretório 'configs' não encontrado")
        return
    
    # Comparação de múltiplas configurações
    if args.compare:
        compare_frameworks(args.compare)
        return
    
    # Execução única
    if not args.config:
        # Tentar usar configuração padrão
        default_configs = ["configs/config.yaml", "config.yaml"]
        config_path = None
        for default_config in default_configs:
            if os.path.exists(default_config):
                config_path = default_config
                break
        
        if config_path is None:
            print("❌ Nenhuma configuração especificada e arquivo padrão não encontrado")
            print("Use --config <caminho> ou --help para mais informações")
            return
    else:
        config_path = args.config
    
    if not os.path.exists(config_path):
        print(f"❌ Arquivo de configuração não encontrado: {config_path}")
        return
    
    # Carregar configuração
    config = load_config(config_path)
    
    # Detectar ou forçar framework
    framework = args.framework if args.framework else detect_framework(config)
    
    # Executar experimento
    print(f"📋 Configuração: {config_path}")
    print(f"🔧 Framework: {framework.upper()}")
    
    try:
        result = train_and_evaluate(config, framework)
        print("\n✅ Experimento concluído com sucesso!")
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()