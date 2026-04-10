import json
import re
from typing import Any, Dict, Optional

from src.rlm.inference import generate_reasoning


class OrchestratorAgent:
    """
    Agente encargado de interpretar la query del usuario y decidir
    qué inputs pasan al resto del sistema.

    Responsabilidades:
    - Extraer ticker o nombre de empresa
    - Extraer perfil de riesgo
    - Extraer horizonte temporal
    - Construir un plan simple de ejecución
    """

    RISK_KEYWORDS = {
        "conservador": "conservador",
        "prudente": "conservador",
        "moderado": "moderado",
        "medio": "moderado",
        "agresivo": "agresivo",
        "arriesgado": "agresivo",
    }

    HORIZON_PATTERNS = [
        r"(\d+\s*(?:mes|meses|año|años))",
        r"(corto plazo)",
        r"(medio plazo)",
        r"(largo plazo)",
        r"(short term)",
        r"(medium term)",
        r"(long term)",
    ]

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def _extract_ticker_heuristic(self, query: str) -> Optional[str]:
        # Caso tipo: Nvidia (NVDA)
        match_parenthesis = re.search(r"\(([A-Z]{1,5})\)", query)
        if match_parenthesis:
            return match_parenthesis.group(1)

        # Tokens en mayúsculas tipo NVDA, AAPL, TSLA
        candidates = re.findall(r"\b[A-Z]{2,5}\b", query)
        blacklist = {"RAG", "LLM", "API", "JSON", "USA", "ETF"}
        candidates = [c for c in candidates if c not in blacklist]
        return candidates[0] if candidates else None

    def _extract_risk_profile(self, query: str) -> str:
        lower = query.lower()
        for key, value in self.RISK_KEYWORDS.items():
            if key in lower:
                return value
        return "moderado"

    def _extract_horizon(self, query: str) -> str:
        lower = query.lower()
        for pattern in self.HORIZON_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                return match.group(1)
        return "12 meses"

    def _extract_company_name(self, query: str) -> Optional[str]:
        """
        Heurística muy simple para casos donde el usuario diga
        'Analiza Nvidia' o 'Compara Microsoft y Amazon'.
        """
        match = re.search(
            r"(?:analiza|invertir en|empresa|acción de|stock de|compra de|sobre)\s+([A-ZÁÉÍÓÚÑ][a-zA-ZÁÉÍÓÚÑáéíóúñ0-9&\-. ]+)",
            query,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(" .,:;")
        return None

    def _llm_parse(self, query: str) -> Dict[str, Any]:
        """
        Fallback opcional con LLM para extraer estructura.
        """
        prompt = f"""
Extrae la información clave de esta consulta financiera y responde SOLO en JSON válido.

Campos:
- company_name: string o null
- ticker: string o null
- risk_profile: "conservador", "moderado" o "agresivo"
- horizon: string corto
- user_goal: string corto

Consulta:
{query}

JSON:
"""
        try:
            raw = generate_reasoning(prompt, self.model, self.tokenizer)
            if "ASSISTANT:" in raw:
                raw = raw.split("ASSISTANT:")[-1].strip()
            raw = raw.replace("<|endoftext|>", "").strip()

            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise ValueError("No se encontró JSON en la salida del modelo.")

            parsed = json.loads(json_match.group(0))
            return {
                "company_name": parsed.get("company_name"),
                "ticker": parsed.get("ticker"),
                "risk_profile": parsed.get("risk_profile", "moderado"),
                "horizon": parsed.get("horizon", "12 meses"),
                "user_goal": parsed.get("user_goal", "análisis de inversión"),
            }
        except Exception:
            return {
                "company_name": self._extract_company_name(query),
                "ticker": self._extract_ticker_heuristic(query),
                "risk_profile": self._extract_risk_profile(query),
                "horizon": self._extract_horizon(query),
                "user_goal": "análisis de inversión",
            }

    def run(self, user_query: str) -> Dict[str, Any]:
        parsed = self._llm_parse(user_query)

        plan = [
            "market_intelligence",
            "recommendation",
            "critic_risk_review",
        ]

        return {
            "user_query": user_query,
            "company_name": parsed.get("company_name"),
            "ticker": parsed.get("ticker"),
            "risk_profile": parsed.get("risk_profile", "moderado"),
            "horizon": parsed.get("horizon", "12 meses"),
            "user_goal": parsed.get("user_goal", "análisis de inversión"),
            "plan": plan,
        }