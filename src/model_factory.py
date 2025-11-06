"""
Factory pattern para criar modelos LambdaMART com diferentes frameworks.
Suporta XGBoost (com IPS) e LightGBM (baseline).
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
import warnings

# Importações condicionais
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost não encontrado. Instale com: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM não encontrado. Instale com: pip install lightgbm")


class ModelFactory:
    """Factory para criar modelos LambdaMART com diferentes frameworks."""
    
    @staticmethod
    def create_model(config: Dict[str, Any]):
        """
        Criar modelo baseado na configuração.
        
        Args:
            config: Configuração do modelo com framework especificado
            
        Returns:
            Instância do modelo (XGBoostLambdaMART ou LightGBMLambdaMART)
        """
        framework = config.get('framework', 'xgboost').lower()
        
        if framework == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost não está instalado. Use: pip install xgboost")
            from .model_xgboost import XGBoostLambdaMART
            return XGBoostLambdaMART(config)
            
        elif framework == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM não está instalado. Use: pip install lightgbm")
            from .model_lightgbm import LightGBMLambdaMART
            return LightGBMLambdaMART(config)
            
        else:
            raise ValueError(f"Framework não suportado: {framework}. Use 'xgboost' ou 'lightgbm'")


class BaseLambdaMART:
    """Classe base para modelos LambdaMART."""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None
        self.feature_importance = None
        self.training_history = None
        
    def train(self, train_features: np.ndarray, train_labels: np.ndarray,
              train_qids: np.ndarray, **kwargs) -> Dict:
        """Treinar o modelo. Deve ser implementado pelas subclasses."""
        raise NotImplementedError("Subclasses devem implementar o método train")
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Fazer predições. Deve ser implementado pelas subclasses."""
        raise NotImplementedError("Subclasses devem implementar o método predict")
        
    def get_feature_importance(self) -> np.ndarray:
        """Obter importância das features. Deve ser implementado pelas subclasses."""
        raise NotImplementedError("Subclasses devem implementar o método get_feature_importance")
        
    def save_model(self, filepath: str):
        """Salvar modelo. Deve ser implementado pelas subclasses."""
        raise NotImplementedError("Subclasses devem implementar o método save_model")
        
    def load_model(self, filepath: str):
        """Carregar modelo. Deve ser implementado pelas subclasses."""
        raise NotImplementedError("Subclasses devem implementar o método load_model")
        
    def get_model_summary(self) -> Dict:
        """Obter resumo do modelo."""
        return {
            'framework': self.__class__.__name__,
            'params': self.params,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history
        }