#!/usr/bin/env python3
"""
Script de Automação - Experimentos LambdaMART
============================================

Executa automaticamente experimentos para todas as combinações de:
- signals: ['click', 'copy']
- methods: ['DCM', 'CCM', 'DBN', 'SDBN', 'SDBN2']

Total: 10 experimentos com hyperparameter tuning (64 trials cada)

Uso: python3 run_all_experiments.py
"""

import os
import subprocess
import sys
import time
from datetime import datetime

# Configurações
SIGNALS = ['click', 'copy']
METHODS = ['DCM', 'CCM', 'DBN', 'SDBN', 'SDBN2']

def check_data_exists(signal, method):
    """Verifica se os dados existem para a combinação."""
    train_path = f"/Users/david/Documents/data/goldenset_paper/svm-format/{signal}/{method}"
    return os.path.exists(train_path)

def run_single_experiment(signal, method, current, total):
    """Executa um único experimento modificando as variáveis de ambiente."""
    
    experiment_name = f"{method}_{signal.upper()}"
    print(f"\n{'='*60}")
    print(f"🚀 EXPERIMENTO {current}/{total}: {experiment_name}")
    print(f"{'='*60}")
    print(f"⏰ Início: {datetime.now().strftime('%H:%M:%S')}")
    
    # Verificar se os dados existem
    if not check_data_exists(signal, method):
        print(f"❌ PULANDO: Dados não encontrados para {signal}/{method}")
        return False, "Dados não encontrados"
    
    # Definir variáveis de ambiente para o experimento
    env = os.environ.copy()
    env['LAMBDAMART_SIGNAL'] = signal
    env['LAMBDAMART_METHOD'] = method
    
    # Executar o script principal
    cmd = ['python3', 'main.py', '--config', 'configs/config_lightgbm.yaml']
    
    start_time = time.time()
    
    try:
        print(f"🏃‍♂️ Executando hyperparameter tuning (64 trials)...")
        print(f"📊 Acompanhe o progresso em tempo real:")
        
        # Executar SEM capturar output para ver logs em tempo real
        result = subprocess.run(
            cmd, 
            env=env, 
            timeout=7200  # 2 horas de timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {experiment_name}: SUCESSO")
            print(f"⏱️  Duração: {duration:.1f}s ({duration/60:.1f}min)")
            return True, f"Concluído em {duration/60:.1f}min"
        else:
            print(f"\n❌ {experiment_name}: FALHA (código {result.returncode})")
            return False, f"Erro (código {result.returncode})"
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {experiment_name}: TIMEOUT (2h)")
        return False, "Timeout após 2 horas"
    except Exception as e:
        print(f"❌ {experiment_name}: EXCEÇÃO - {e}")
        return False, f"Exceção: {str(e)}"

def main():
    """Executa todos os experimentos."""
    print("🤖 AUTOMAÇÃO DE EXPERIMENTOS LAMBDAMART")
    print("=" * 50)
    print(f"📅 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"� Framework: LightGBM")
    print(f"⚙️  Hyperparameter Tuning: 64 trials por experimento")
    print(f"📊 Signals: {SIGNALS}")
    print(f"🔬 Methods: {METHODS}")
    
    # Verificar quais combinações existem
    valid_combinations = []
    for signal in SIGNALS:
        for method in METHODS:
            if check_data_exists(signal, method):
                valid_combinations.append((signal, method))
                print(f"✅ {signal}/{method}")
            else:
                print(f"❌ {signal}/{method} (dados não encontrados)")
    
    total_experiments = len(valid_combinations)
    print(f"\n🎯 Experimentos válidos: {total_experiments}")
    
    if total_experiments == 0:
        print("❌ ERRO: Nenhuma combinação de dados válida!")
        return 1
    
    estimated_time = total_experiments * 30  # ~30min por experimento estimado
    print(f"⏱️  Tempo estimado: ~{estimated_time}min ({estimated_time/60:.1f}h)")
    
    input("\n⏳ Pressione ENTER para iniciar ou Ctrl+C para cancelar...")
    
    # Executar experimentos
    results = []
    total_start = time.time()
    
    for i, (signal, method) in enumerate(valid_combinations, 1):
        success, message = run_single_experiment(signal, method, i, total_experiments)
        results.append((signal, method, success, message))
        
        # Pequena pausa entre experimentos
        if i < total_experiments:
            print("⏸️  Pausa de 10 segundos...")
            time.sleep(10)
    
    total_duration = time.time() - total_start
    
    # Relatório final
    print(f"\n{'='*60}")
    print("📋 RELATÓRIO FINAL")
    print(f"{'='*60}")
    print(f"⏱️  Tempo total: {total_duration/60:.1f}min ({total_duration/3600:.1f}h)")
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
                print(f"   ✅ {method}_{signal.upper()}: {message}")
    
    if failed > 0:
        print(f"\n❌ EXPERIMENTOS FALHADOS:")
        for signal, method, success, message in results:
            if not success:
                print(f"   ❌ {method}_{signal.upper()}: {message}")
    
    print(f"\n🔬 Para visualizar todos os resultados:")
    print(f"   mlflow ui")
    print(f"   http://localhost:5000")
    print(f"\n📊 Cada experimento tem seu próprio experimento no MLflow:")
    for signal, method, success, _ in results:
        if success:
            print(f"   - LambdaMART_LightGBM_{method}_{signal.title()}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 AUTOMAÇÃO INTERROMPIDA PELO USUÁRIO")
        print("✋ Experimentos em andamento foram cancelados.")
        sys.exit(130)