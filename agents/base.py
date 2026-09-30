from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def report(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives global sensory & forecast state and returns a structured agent report.
        """
        pass
