import json
import re
from typing import Any, Dict

from src.rlm.inference import generate_reasoning


def recommendation_fallback(text: str) -> str:
    lower = text.lower()
    if "desfavorable" in lower:
        return "desfavorable"
    if "favorable" in lower:
        return "favorable"
    return "neutral"


class CriticRiskAgent:
    """
    Agente crítico. No vuelve a hacer la recomendación desde cero.
    Revisa si la tesis:
    - está apoyada en hechos
    - cubre riesgos
    - usa un lenguaje prudente
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def _build_prompt(
        self,
        market_report: Dict[str, Any],
        recommendation_report: Dict[str, Any],
        risk_profile: str,
        horizon: str,
        user_query: str,
    ) -> str:
        return f"""
Actúas como revisor crítico de una recomendación financiera.
Tu función no es rehacer todo el análisis, sino revisar si está bien fundamentado.

Consulta original:
{user_query}

Perfil del usuario: {risk_profile}
Horizonte temporal: {horizon}

INFORME DE MERCADO:
{market_report}

RECOMENDACIÓN PRELIMINAR:
{recommendation_report}

Responde SOLO en JSON válido con esta estructura:
{{
  "grounded_in_facts": true,
  "missing_risks": ["string"],
  "consistency_issues": ["string"],
  "language_adjustments": ["string"],
  "final_recommendation": "favorable|neutral|desfavorable",
  "final_answer": "string"
}}

Reglas:
- grounded_in_facts debe ser true o false
- si no hay problemas, devuelve listas vacías
- final_answer debe ser prudente, clara y breve
- no añadas texto fuera del JSON
"""

    def _parse_json(self, raw_output: str) -> Dict[str, Any]:
        try:
            if "ASSISTANT:" in raw_output:
                raw_output = raw_output.split("ASSISTANT:")[-1].strip()
            raw_output = raw_output.replace("<|endoftext|>", "").strip()

            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if not json_match:
                raise ValueError("No se encontró JSON en la salida.")

            parsed = json.loads(json_match.group(0))
            return {
                "grounded_in_facts": parsed.get("grounded_in_facts", False),
                "missing_risks": parsed.get("missing_risks", []),
                "consistency_issues": parsed.get("consistency_issues", []),
                "language_adjustments": parsed.get("language_adjustments", []),
                "final_recommendation": parsed.get("final_recommendation", "neutral"),
                "final_answer": parsed.get("final_answer", ""),
                "raw_output": raw_output,
            }
        except Exception:
            return {
                "grounded_in_facts": False,
                "missing_risks": [],
                "consistency_issues": ["No se pudo parsear la salida del critic agent."],
                "language_adjustments": [],
                "final_recommendation": recommendation_fallback(raw_output),
                "final_answer": raw_output[:1000],
                "raw_output": raw_output,
            }

    def run(
        self,
        market_report: Dict[str, Any],
        recommendation_report: Dict[str, Any],
        risk_profile: str,
        horizon: str,
        user_query: str,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(
            market_report=market_report,
            recommendation_report=recommendation_report,
            risk_profile=risk_profile,
            horizon=horizon,
            user_query=user_query,
        )
        raw_output = generate_reasoning(prompt, self.model, self.tokenizer)
        return self._parse_json(raw_output)