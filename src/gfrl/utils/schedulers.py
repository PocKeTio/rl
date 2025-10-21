"""
Schedulers pour annealing de paramètres (entropy, learning rate, etc.).
"""

from typing import Literal


class AnnealingScheduler:
    """
    Scheduler pour annealing linéaire ou exponentiel d'un paramètre.
    """
    
    def __init__(
        self,
        start_value: float,
        end_value: float,
        total_steps: int,
        strategy: Literal["linear", "exponential"] = "linear",
    ):
        """
        Args:
            start_value: Valeur initiale
            end_value: Valeur finale
            total_steps: Nombre de steps pour l'annealing
            strategy: Stratégie d'annealing ("linear" ou "exponential")
        """
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps
        self.strategy = strategy
        self.current_step = 0
        
    def step(self) -> float:
        """
        Avance d'un step et retourne la valeur actuelle.
        
        Returns:
            Valeur actuelle du paramètre
        """
        if self.current_step >= self.total_steps:
            return self.end_value
        
        progress = self.current_step / self.total_steps
        
        if self.strategy == "linear":
            value = self.start_value + (self.end_value - self.start_value) * progress
        elif self.strategy == "exponential":
            # Décroissance exponentielle
            value = self.start_value * (self.end_value / self.start_value) ** progress
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.current_step += 1
        return value
    
    def get_value(self) -> float:
        """
        Retourne la valeur actuelle sans avancer.
        
        Returns:
            Valeur actuelle du paramètre
        """
        if self.current_step >= self.total_steps:
            return self.end_value
        
        progress = self.current_step / self.total_steps
        
        if self.strategy == "linear":
            return self.start_value + (self.end_value - self.start_value) * progress
        elif self.strategy == "exponential":
            return self.start_value * (self.end_value / self.start_value) ** progress
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def reset(self):
        """Reset le scheduler."""
        self.current_step = 0
