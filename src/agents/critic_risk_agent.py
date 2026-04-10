import json
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
    - tiene evidencia suficiente
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
        compact_market = {
            "company_name": market_report.get("company_name"),
            "ticker": market_report.get("ticker"),
            "summary": market_report.get("summary"),
            "price_data": market_report.get("price_data"),
            "fundamentals_data": market_report.get("fundamentals_data"),
            "events_data": market_report.get("events_data"),
            "has_minimum_evidence": market_report.get("has_minimum_evidence", False),
        }

        compact_recommendation = {
            "thesis": recommendation_report.get("thesis"),
            "strengths": recommendation_report.get("strengths"),
            "risks": recommendation_report.get("risks"),
            "scenarios": recommendation_report.get("scenarios"),
            "preliminary_recommendation": recommendation_report.get("preliminary_recommendation"),
            "confidence": recommendation_report.get("confidence"),
        }

        return f"""
Actúas como revisor crítico de una recomendación financiera.
Tu función no es rehacer todo el análisis, sino revisar si está bien fundamentado.

Consulta original:
{user_query}

Perfil del usuario: {risk_profile}
Horizonte temporal: {horizon}

INFORME DE MERCADO:
{compact_market}

RECOMENDACIÓN PRELIMINAR:
{compact_recommendation}

Responde SOLO entre BEGIN_JSON y END_JSON con un JSON válido de esta estructura:

BEGIN_JSON
{{
  "enough_evidence": true,
  "grounded_in_facts": true,
  "missing_risks": ["string"],
  "consistency_issues": ["string"],
  "language_adjustments": ["string"],
  "final_recommendation": "favorable|neutral|desfavorable",
  "final_answer": "string"
}}
END_JSON

Reglas:
- enough_evidence debe indicar si existe base suficiente para emitir una recomendación
- si no hay suficiente evidencia, final_answer debe decirlo explícitamente y ser prudente
- grounded_in_facts debe ser true o false
- si no hay problemas, devuelve listas vacías
- no añadas texto fuera del bloque JSON
"""

    def _extract_json_block(self, text: str) -> str | None:
        if "BEGIN_JSON" in text and "END_JSON" in text:
            start = text.find("BEGIN_JSON") + len("BEGIN_JSON")
            end = text.find("END_JSON", start)
            if end != -1:
                return text[start:end].strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start:end + 1].strip()

    def _parse_json(self, raw_output: str) -> Dict[str, Any]:
        cleaned_output = raw_output

        try:
            if "ASSISTANT:" in cleaned_output:
                cleaned_output = cleaned_output.split("ASSISTANT:")[-1].strip()
            cleaned_output = cleaned_output.replace("<|endoftext|>", "").strip()

            json_block = self._extract_json_block(cleaned_output)
            if not json_block:
                raise ValueError("No se encontró JSON en la salida.")

            parsed = json.loads(json_block)
            return {
                "enough_evidence": parsed.get("enough_evidence", False),
                "grounded_in_facts": parsed.get("grounded_in_facts", False),
                "missing_risks": parsed.get("missing_risks", []),
                "consistency_issues": parsed.get("consistency_issues", []),
                "language_adjustments": parsed.get("language_adjustments", []),
                "final_recommendation": parsed.get("final_recommendation", "neutral"),
                "final_answer": parsed.get("final_answer", ""),
                "raw_output": cleaned_output,
            }
        except Exception:
            return {
                "enough_evidence": False,
                "grounded_in_facts": False,
                "missing_risks": [],
                "consistency_issues": ["No se pudo parsear la salida del critic agent."],
                "language_adjustments": [],
                "final_recommendation": recommendation_fallback(cleaned_output),
                "final_answer": cleaned_output[:1000],
                "raw_output": cleaned_output,
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