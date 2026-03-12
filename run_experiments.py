#!/usr/bin/env python3
"""
Automação de Experimentos LambdaMART
====================================

Script para executar automaticamente experimentos LambdaMART com LightGBM
em todas as combinações de signals e methods.

Combinações executadas:
- signals: ['click', 'copy']  
- methods: ['DCM', 'CCM', 'DBN', 'SDBN', 'SDBN2']
- Total: 2 x 5 = 10 experimentos

Cada combinação será salva como um experimento separado no MLflow.
"""

import os
import sys
import yaml
import subprocess
import time
from datetime import datetime
from typing import List, Dict, Tuple

# Configurações
SIGNALS = ['click', 'copy']
METHODS = ['DCM', 'CCM', 'DBN', 'SDBN', 'SDBN2']
BASE_CONFIG_PATH = 'configs/config_lightgbm.yaml'
TEMP_CONFIG_PATH = 'configs/config_lightgbm_auto.yaml'
BASE_DATA_PATH = '/Users/david/Documents/data/goldenset_paper/svm-format'
TEST_FILE = '/Users/david/Documents/data/goldenset_paper/colecao_manual.txt'


def load_base_config() -> Dict:
    """Carrega a configuração base do LightGBM."""
    with open(BASE_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_experiment_config(signal: str, method: str, base_config: Dict) -> Dict:
    """
    Cria uma configuração específica para um experimento.
    
    Args:
        signal: Tipo de signal ('click' ou 'copy')
        method: Método usado ('DCM', 'CCM', etc.)
        base_config: Configuração base
        
    Returns:
        Configuração modificada para o experimento
    """
    config = base_config.copy()
    
    # Atualizar caminhos dos dados
    train_dir = f"{BASE_DATA_PATH}/{signal}/{method}"
    config['data']['train_dir'] = train_dir
    config['data']['test_dir'] = TEST_FILE
    
    # Atualizar nome do experimento MLflow
    experiment_name = f"LambdaMART_LightGBM_{method}_{signal.title()}"
    config['mlflow']['experiment_name'] = experiment_name
    
    return config


def save_temp_config(config: Dict) -> None:
    """Salva uma configuração temporária."""
    with open(TEMP_CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)


def run_experiment(signal: str, method: str) -> Tuple[bool, str]:
    """
    Executa um experimento específico.
    
    Args:
        signal: Tipo de signal
        method: Método usado
        
    Returns:
        Tuple de (sucesso, mensagem)
    """
    experiment_name = f"{method}_{signal.title()}"
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO EXPERIMENTO: {experiment_name}")
    print(f"{'='*60}")
    print(f"📊 Signal: {signal}")
    print(f"🔧 Method: {method}")
    print(f"📁 Train Dir: {BASE_DATA_PATH}/{signal}/{method}")
    print(f"⏰ Hora: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        # Carregar configuração base
        base_config = load_base_config()
        
        # Criar configuração específica do experimento
        experiment_config = create_experiment_config(signal, method, base_config)
        
        # Salvar configuração temporária
        save_temp_config(experiment_config)
        
        # Executar o experimento
        cmd = [sys.executable, 'main.py', '--config', TEMP_CONFIG_PATH]
        
        print(f"🏃‍♂️ Executando: {' '.join(cmd)}")
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hora de timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCESSO: {experiment_name}")
            print(f"⏱️  Duração: {duration:.1f}s ({duration/60:.1f}min)")
            
            # Extrair métricas do output (se possível)
            if "NDCG@10" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "ndcg@10" in line.lower():
                        print(f"📊 {line.strip()}")
                        break
            
            return True, f"Experimento concluído em {duration:.1f}s"
        else:
            print(f"❌ ERRO: {experiment_name}")
            print(f"💥 Código de saída: {result.returncode}")
            print(f"📝 Stderr: {result.stderr[:500]}")
            return False, f"Erro: {result.stderr[:200]}"
            
    except subprocess.TimeoutExpired:
        return False, "Timeout: experimento excedeu 1 hora"
    except Exception as e:
        return False, f"Exceção: {str(e)}"
    finally:
        # Limpar arquivo temporário
        if os.path.exists(TEMP_CONFIG_PATH):
            os.remove(TEMP_CONFIG_PATH)


def validate_data_paths() -> List[Tuple[str, str]]:
    """
    Valida quais combinações de dados existem.
    
    Returns:
        Lista de combinações válidas (signal, method)
    """
    valid_combinations = []
    missing_combinations = []
    
    print("🔍 VALIDANDO CAMINHOS DOS DADOS...")
    print(f"📁 Base path: {BASE_DATA_PATH}")
    
    for signal in SIGNALS:
        for method in METHODS:
            train_path = f"{BASE_DATA_PATH}/{signal}/{method}"
            if os.path.exists(train_path):
                valid_combinations.append((signal, method))
                print(f"✅ {signal}/{method}")
            else:
                missing_combinations.append((signal, method))
                print(f"❌ {signal}/{method} (não encontrado)")
    
    print(f"\n📊 RESUMO DA VALIDAÇÃO:")
    print(f"✅ Combinações válidas: {len(valid_combinations)}")
    print(f"❌ Combinações faltando: {len(missing_combinations)}")
    
    if missing_combinations:
        print(f"\n⚠️  CAMINHOS FALTANDO:")
        for signal, method in missing_combinations:
            print(f"   - {BASE_DATA_PATH}/{signal}/{method}")
    
    return valid_combinations


def main():
    """Função principal da automação."""
    print("🤖 AUTOMAÇÃO DE EXPERIMENTOS LAMBDAMART")
    print("=" * 50)
    print(f"📅 Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Framework: LightGBM")
    print(f"⚙️  Hyperparameter Tuning: Habilitado")
    print(f"📊 Signals: {SIGNALS}")
    print(f"🔬 Methods: {METHODS}")
    print(f"🎯 Total de experimentos: {len(SIGNALS) * len(METHODS)}")
    
    # Validar caminhos dos dados
    valid_combinations = validate_data_paths()
    
    if not valid_combinations:
        print("\n❌ ERRO: Nenhuma combinação de dados válida encontrada!")
        print("Verifique se os caminhos dos dados estão corretos.")
        return 1
    
    print(f"\n🚀 INICIANDO {len(valid_combinations)} EXPERIMENTOS...")
    
    # Executar experimentos
    results = []
    total_start_time = time.time()
    
    for i, (signal, method) in enumerate(valid_combinations, 1):
        print(f"\n🎯 PROGRESSO: {i}/{len(valid_combinations)}")
        success, message = run_experiment(signal, method)
        results.append((signal, method, success, message))
        
        if not success:
            print(f"⚠️  Experimento falhou, continuando com próximo...")
        
        # Pequena pausa entre experimentos
        if i < len(valid_combinations):
            print("⏸️  Pausa de 5 segundos...")
            time.sleep(5)
    
    total_duration = time.time() - total_start_time
    
    # Relatório final
    print(f"\n{'='*60}")
    print("📋 RELATÓRIO FINAL")
    print(f"{'='*60}")
    print(f"⏱️  Tempo total: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"🎯 Experimentos executados: {len(results)}")
    
    successful = sum(1 for _, _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"✅ Sucessos: {successful}")
    print(f"❌ Falhas: {failed}")
    print(f"📊 Taxa de sucesso: {successful/len(results)*100:.1f}%")
    
    if successful > 0:
        print(f"\n✅ EXPERIMENTOS BEM-SUCEDIDOS:")
        for signal, method, success, message in results:
            if success:
                print(f"   ✅ {method}_{signal.title()}: {message}")
    
    if failed > 0:
        print(f"\n❌ EXPERIMENTOS FALHADOS:")
        for signal, method, success, message in results:
            if not success:
                print(f"   ❌ {method}_{signal.title()}: {message}")
    
    print(f"\n🔬 Para visualizar os resultados:")
    print(f"   mlflow ui")
    print(f"   http://localhost:5000")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)