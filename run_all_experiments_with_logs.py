#!/usr/bin/env python3
"""
Script de Automação com Logs - Experimentos LambdaMART
=====================================================

Igual ao run_all_experiments.py mas salva logs detalhados de cada experimento.

Uso: python3 run_all_experiments_with_logs.py
"""

import os
import subprocess
import sys
import time
from datetime import datetime

# Configurações
SIGNALS = ['click', 'copy']
METHODS = ['DCM', 'CCM', 'DBN', 'SDBN', 'SDBN2']
LOGS_DIR = "experiment_logs"

def setup_logs_directory():
    """Cria diretório para logs se não existir."""
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

def check_data_exists(signal, method):
    """Verifica se os dados existem para a combinação."""
    train_path = f"/Users/david/Documents/data/goldenset_paper/svm-format/{signal}/{method}"
    return os.path.exists(train_path)

def run_single_experiment_with_logs(signal, method, current, total):
    """Executa um único experimento e salva logs."""
    
    experiment_name = f"{method}_{signal.upper()}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{LOGS_DIR}/{experiment_name}_{timestamp}.log"
    
    print(f"\n{'='*60}")
    print(f"🚀 EXPERIMENTO {current}/{total}: {experiment_name}")
    print(f"{'='*60}")
    print(f"⏰ Início: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📝 Log: {log_file}")
    
    # Verificar se os dados existem
    if not check_data_exists(signal, method):
        print(f"❌ PULANDO: Dados não encontrados para {signal}/{method}")
        return False, "Dados não encontrados"
    
    # Definir variáveis de ambiente
    env = os.environ.copy()
    env['LAMBDAMART_SIGNAL'] = signal
    env['LAMBDAMART_METHOD'] = method
    
    # Executar o script principal
    cmd = ['python3', 'main.py', '--config', 'configs/config_lightgbm.yaml']
    
    start_time = time.time()
    
    try:
        print(f"🏃‍♂️ Executando hyperparameter tuning (64 trials)...")
        print(f"📊 Acompanhe em tempo real ou veja logs: tail -f {log_file}")
        
        # Executar salvando logs em arquivo
        with open(log_file, 'w') as f:
            # Escrever cabeçalho no log
            f.write(f"=== EXPERIMENTO: {experiment_name} ===\n")
            f.write(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Signal: {signal}, Method: {method}\n")
            f.write(f"Comando: {' '.join(cmd)}\n")
            f.write("=" * 50 + "\n\n")
            f.flush()
            
            # Executar processo redirecionando output para arquivo E console
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Ler output linha por linha e escrever no arquivo E console
            for line in process.stdout:
                # Escrever no arquivo de log
                f.write(line)
                f.flush()
                
                # Mostrar progresso no console (apenas algumas linhas importantes)
                if any(keyword in line.lower() for keyword in [
                    'trial', 'melhor score', 'training completed', 'ndcg@10', 'erro'
                ]):
                    print(f"  {line.strip()}")
            
            process.wait(timeout=7200)  # 2 horas de timeout
            
            duration = time.time() - start_time
            
            # Escrever rodapé no log
            f.write(f"\n" + "=" * 50)
            f.write(f"\nFim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            f.write(f"\nDuração: {duration:.1f}s ({duration/60:.1f}min)")
            f.write(f"\nCódigo de saída: {process.returncode}")
        
        if process.returncode == 0:
            print(f"✅ {experiment_name}: SUCESSO")
            print(f"⏱️  Duração: {duration:.1f}s ({duration/60:.1f}min)")
            print(f"📝 Log salvo: {log_file}")
            return True, f"Concluído em {duration/60:.1f}min"
        else:
            print(f"❌ {experiment_name}: FALHA (código {process.returncode})")
            print(f"📝 Log com detalhes: {log_file}")
            return False, f"Erro (código {process.returncode})"
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {experiment_name}: TIMEOUT (2h)")
        return False, "Timeout após 2 horas"
    except Exception as e:
        print(f"❌ {experiment_name}: EXCEÇÃO - {e}")
        return False, f"Exceção: {str(e)}"

def main():
    """Executa todos os experimentos com logs."""
    print("🤖 AUTOMAÇÃO DE EXPERIMENTOS LAMBDAMART (COM LOGS)")
    print("=" * 55)
    
    # Configurar diretório de logs
    setup_logs_directory()
    
    print(f"📅 Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Framework: LightGBM")
    print(f"⚙️  Hyperparameter Tuning: 64 trials por experimento")
    print(f"📝 Logs salvos em: {LOGS_DIR}/")
    print(f"📊 Signals: {SIGNALS}")
    print(f"🔬 Methods: {METHODS}")
    
    # Verificar combinações válidas
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
    
    estimated_time = total_experiments * 30
    print(f"⏱️  Tempo estimado: ~{estimated_time}min ({estimated_time/60:.1f}h)")
    print(f"📝 Para acompanhar logs em tempo real:")
    print(f"   tail -f {LOGS_DIR}/<experimento>_*.log")
    
    input("\n⏳ Pressione ENTER para iniciar ou Ctrl+C para cancelar...")
    
    # Executar experimentos
    results = []
    total_start = time.time()
    
    for i, (signal, method) in enumerate(valid_combinations, 1):
        success, message = run_single_experiment_with_logs(signal, method, i, total_experiments)
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
    print(f"📝 Logs salvos em: {LOGS_DIR}/")
    
    successful = sum(1 for _, _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"✅ Sucessos: {successful}")
    print(f"❌ Falhas: {failed}")
    print(f"📊 Taxa de sucesso: {successful/len(results)*100:.1f}%")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 AUTOMAÇÃO INTERROMPIDA PELO USUÁRIO")
        sys.exit(130)