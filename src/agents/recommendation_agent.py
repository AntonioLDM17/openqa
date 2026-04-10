import json
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

    def _compress_market_report(self, market_report: Dict[str, Any]) -> Dict[str, Any]:
        rag_context = market_report.get("rag_context", [])[:2]
        compressed_rag = [
            {
                "label": ctx.get("label"),
                "distance": ctx.get("distance"),
                "text": ctx.get("text", "")[:300],
            }
            for ctx in rag_context
        ]

        external_context = market_report.get("external_context")
        if isinstance(external_context, str):
            external_context = external_context[:800]

        return {
            "company_name": market_report.get("company_name"),
            "ticker": market_report.get("ticker"),
            "price_data": market_report.get("price_data"),
            "fundamentals_data": market_report.get("fundamentals_data"),
            "events_data": market_report.get("events_data"),
            "external_context": external_context,
            "rag_context": compressed_rag,
            "summary": market_report.get("summary"),
            "has_minimum_evidence": market_report.get("has_minimum_evidence", False),
        }

    def _build_prompt(
        self,
        market_report: Dict[str, Any],
        risk_profile: str,
        horizon: str,
        user_query: str,
    ) -> str:
        compact_report = self._compress_market_report(market_report)

        return f"""
Eres un analista financiero especializado en construir tesis de inversión razonadas.
Debes basarte solo en la información proporcionada.
No inventes datos. Si falta evidencia, dilo explícitamente.
No uses lenguaje excesivamente tajante si la evidencia es limitada.

Consulta del usuario:
{user_query}

Perfil de riesgo del usuario: {risk_profile}
Horizonte temporal: {horizon}

INFORME DE MERCADO:
{compact_report}

Responde SOLO entre las etiquetas BEGIN_JSON y END_JSON con un JSON válido de esta estructura exacta:

BEGIN_JSON
{{
  "thesis": "string",
  "strengths": ["string", "string"],
  "risks": ["string", "string"],
  "scenarios": ["string", "string"],
  "preliminary_recommendation": "favorable|neutral|desfavorable",
  "confidence": "baja|media|alta"
}}
END_JSON
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