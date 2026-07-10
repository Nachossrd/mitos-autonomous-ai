"""
Módulo Cerebro: motor LLM local + memoria vectorial persistente.

Este paquete materializa el loop autónomo:
    Observe  ->  Think  ->  Act  ->  Learn

Sin APIs externas, sin datos en la nube. Toda la inferencia y todo el
estado vectorial viven en el host. La privacidad y la disponibilidad del
sistema son propiedades emergentes de no depender de proveedores remotos.
"""

from .llm_engine import LLMEngine
from .memory import Memory, MemorySystem

# AgentState/AutonomousAgent del CLI legacy ya no se re-exportan aquí —
# el modo --voice no los usa. Quien aún quiera el CLI brain
# (run_brain.ps1) importa directo: `from src.brain.agent import ...`.

__all__ = ["LLMEngine", "Memory", "MemorySystem"]
