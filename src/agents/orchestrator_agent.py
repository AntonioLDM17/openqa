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

    STOP_WORDS = {
        "y", "o", "para", "con", "sin", "de", "del", "ahora", "hoy",
        "mañana", "porque", "si", "tendría", "sentido", "entrar",
        "invertir", "perfil", "moderado", "conservador", "agresivo",
        "mes", "meses", "año", "años", "plazo"
    }

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def _extract_ticker_heuristic(self, query: str) -> Optional[str]:
        match_parenthesis = re.search(r"\(([A-Z]{1,5})\)", query)
        if match_parenthesis:
            return match_parenthesis.group(1)

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

    def _clean_company_candidate(self, candidate: str) -> Optional[str]:
        candidate = candidate.strip(" .,:;¿?¡!()[]{}\"'")
        candidate = re.split(
            r"\b(?:y|o|para|con|sin|porque|si|que|a|en|ahora|hoy|mañana)\b",
            candidate,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        words = candidate.split()
        cleaned_words = []
        for w in words:
            wl = w.lower().strip(" .,:;")
            if wl in self.STOP_WORDS:
                break
            cleaned_words.append(w.strip(" .,:;"))

        candidate = " ".join(cleaned_words).strip()

        if not candidate:
            return None

        if len(candidate.split()) > 5:
            candidate = " ".join(candidate.split()[:5]).strip()

        return candidate or None

    def _extract_company_name(self, query: str) -> Optional[str]:
        """
        Heurística mejorada para casos tipo:
        - Analiza Nvidia
        - Compara Microsoft y Amazon
        - Tiene sentido invertir en Tesla ahora
        """
        patterns = [
            r"(?:analiza|estudia|revisa)\s+([A-ZÁÉÍÓÚÑ][a-zA-ZÁÉÍÓÚÑáéíóúñ0-9&\-. ]+)",
            r"(?:invertir en|entrada en|comprar)\s+([A-ZÁÉÍÓÚÑ][a-zA-ZÁÉÍÓÚÑáéíóúñ0-9&\-. ]+)",
            r"(?:empresa|acción de|stock de)\s+([A-ZÁÉÍÓÚÑ][a-zA-ZÁÉÍÓÚÑáéíóúñ0-9&\-. ]+)",
            r"(?:sobre)\s+([A-ZÁÉÍÓÚÑ][a-zA-ZÁÉÍÓÚÑáéíóúñ0-9&\-. ]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                cleaned = self._clean_company_candidate(match.group(1))
                if cleaned:
                    return cleaned

        return None

    def _extract_json_block(self, text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start:end + 1]

    def _llm_parse(self, query: str) -> Dict[str, Any]:
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

Devuelve SOLO un objeto JSON.
"""
        try:
            raw = generate_reasoning(prompt, self.model, self.tokenizer)
            if "ASSISTANT:" in raw:
                raw = raw.split("ASSISTANT:")[-1].strip()
            raw = raw.replace("<|endoftext|>", "").strip()

            json_block = self._extract_json_block(raw)
            if not json_block:
                raise ValueError("No se encontró JSON en la salida del modelo.")

            parsed = json.loads(json_block)

            company_name = parsed.get("company_name")
            if isinstance(company_name, str):
                company_name = self._clean_company_candidate(company_name)

            return {
                "company_name": company_name,
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