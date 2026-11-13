"""3DFUN AI Reconstruction Engine (Public API).

Course: 3D Computer Vision (Autumn Semester 2026)
Institution: Indian Institute of Information Technology (IIIT), Dharwad
Supervisor: Dr. Shirshendu Layek, Assistant Professor, Dept. of Data Science & AI

Student Research Team:
  - J Ganesh (23BDS024) — Data Science and Artificial Intelligence
  - M Jagadeshwar (23BDS033) — Data Science and Artificial Intelligence
  - B Varshith (23BDS011) — Data Science and Artificial Intelligence
  - Pranav Kashyap (23BCS098) — Computer Science and Engineering
  - C Sai Aravind (23BDS075) — Data Science and Artificial Intelligence
"""
from .config import PipelineConfig, preset_config, QUALITY_PRESETS
from .pipeline import STAGES, ReconstructionResult, reconstruct

__all__ = ["PipelineConfig", "preset_config", "QUALITY_PRESETS", "STAGES",
           "ReconstructionResult", "reconstruct"]
__version__ = "1.0.0"
