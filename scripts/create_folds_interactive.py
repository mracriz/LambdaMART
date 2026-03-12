#!/usr/bin/env python3
"""
Script simplificado para criar folds estratificados.
Uso interativo mais amigável.
"""

import os
import sys
from pathlib import Path

def main():
    print("=== Gerador de Folds Estratificados para LambdaMART ===\n")
    
    # Obter diretório base do projeto
    base_dir = Path(__file__).parent.parent
    
    # Solicitar arquivo CSV de entrada
    print("Arquivos CSV disponíveis em data/:")
    data_dir = base_dir / "data"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file}")
    
    print(f"  {len(csv_files) + 1}. Outro arquivo (digitar caminho completo)")
    
    choice = input(f"\nEscolha o arquivo CSV (1-{len(csv_files) + 1}): ").strip()
    
    try:
        choice_num = int(choice)
        if 1 <= choice_num <= len(csv_files):
            input_csv = data_dir / csv_files[choice_num - 1]
        elif choice_num == len(csv_files) + 1:
            input_csv = input("Digite o caminho completo para o arquivo CSV: ").strip()
            input_csv = Path(input_csv)
        else:
            print("Escolha inválida!")
            return 1
    except ValueError:
        print("Entrada inválida!")
        return 1
    
    if not input_csv.exists():
        print(f"Arquivo não encontrado: {input_csv}")
        return 1
    
    # Solicitar diretório de saída
    output_dir = input("\nDigite o caminho de saída (ex: /Users/david/Documents/data/goldenset_paper/svm-format/manual): ").strip()
    output_dir = Path(output_dir)
    
    # Solicitar número de folds
    n_folds = input("Número de folds (padrão: 5): ").strip()
    if not n_folds:
        n_folds = "5"
    
    try:
        n_folds = int(n_folds)
        if n_folds < 2:
            print("Número de folds deve ser pelo menos 2!")
            return 1
    except ValueError:
        print("Número de folds inválido!")
        return 1
    
    # Confirmar parâmetros
    print(f"\n=== Configuração ===")
    print(f"Arquivo de entrada: {input_csv}")
    print(f"Diretório de saída: {output_dir}")
    print(f"Número de folds: {n_folds}")
    
    confirm = input("\nProsseguir? (s/n): ").strip().lower()
    if confirm not in ['s', 'sim', 'y', 'yes']:
        print("Operação cancelada.")
        return 0
    
    # Executar script principal
    main_script = base_dir / "scripts" / "create_stratified_folds.py"
    feature_map = base_dir / "data" / "feature_map.json"
    
    cmd = f'python "{main_script}" --input "{input_csv}" --output "{output_dir}" --n_folds {n_folds} --feature_map "{feature_map}"'
    
    print(f"\nExecutando: {cmd}\n")
    return os.system(cmd)

if __name__ == "__main__":
    exit(main())