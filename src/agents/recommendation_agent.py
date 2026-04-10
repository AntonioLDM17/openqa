import json
import re
from typing import Any, Dict

from src.models.fin_model import generate_financial_reasoning


class RecommendationAgent:
    """
    Agente encargado de convertir el market_report en una tesis de inversión.

    Aquí es donde tiene sentido conectar Fin-R1.
    """

    def __init__(self, fin_model: Any, fin_tokenizer: Any):
        self.fin_model = fin_model
        self.fin_tokenizer = fin_tokenizer

    def _build_prompt(
        self,
        market_report: Dict[str, Any],
        risk_profile: str,
        horizon: str,
        user_query: str,
    ) -> str:
        return f"""
Eres un analista financiero especializado en construir tesis de inversión razonadas.
Debes basarte solo en la información proporcionada.
No inventes datos. Si falta evidencia, dilo explícitamente.

Consulta del usuario:
{user_query}

Perfil de riesgo del usuario: {risk_profile}
Horizonte temporal: {horizon}

INFORME DE MERCADO:
- company_name: {market_report.get("company_name")}
- ticker: {market_report.get("ticker")}
- price_data: {market_report.get("price_data")}
- fundamentals_data: {market_report.get("fundamentals_data")}
- events_data: {market_report.get("events_data")}
- external_context: {market_report.get("external_context")}
- rag_context: {market_report.get("rag_context")}
- summary: {market_report.get("summary")}

Genera una recomendación preliminar en JSON válido con esta estructura exacta:
{{
  "thesis": "string",
  "strengths": ["string", "string"],
  "risks": ["string", "string"],
  "scenarios": ["string", "string"],
  "preliminary_recommendation": "favorable|neutral|desfavorable",
  "confidence": "baja|media|alta"
}}

No añadas texto fuera del JSON.
"""

    def _parse_json(self, raw_output: str) -> Dict[str, Any]:
        cleaned_output = raw_output

        try:
            if "ASSISTANT:" in cleaned_output:
                cleaned_output = cleaned_output.split("ASSISTANT:")[-1].strip()
            cleaned_output = cleaned_output.replace("<|endoftext|>", "").strip()

            json_match = re.search(r"\{.*\}", cleaned_output, re.DOTALL)
            if not json_match:
                raise ValueError("No se encontró JSON en la salida.")

            parsed = json.loads(json_match.group(0))
            return {
                "thesis": parsed.get("thesis", ""),
                "strengths": parsed.get("strengths", []),
                "risks": parsed.get("risks", []),
                "scenarios": parsed.get("scenarios", []),
                "preliminary_recommendation": parsed.get(
                    "preliminary_recommendation",
                    "neutral"
                ),
                "confidence": parsed.get("confidence", "media"),
                "raw_output": cleaned_output,
            }
        except Exception:
            return {
                "thesis": cleaned_output[:1000],
                "strengths": [],
                "risks": [],
                "scenarios": [],
                "preliminary_recommendation": "neutral",
                "confidence": "media",
                "raw_output": cleaned_output,
            }

    def run(
        self,
        market_report: Dict[str, Any],
        risk_profile: str,
        horizon: str,
        user_query: str,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(
            market_report=market_report,
            risk_profile=risk_profile,
            horizon=horizon,
            user_query=user_query,
        )

        raw_output = generate_financial_reasoning(
            prompt,
            self.fin_model,
            self.fin_tokenizer,
        )

        return self._parse_json(raw_output)