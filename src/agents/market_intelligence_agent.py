import re
from typing import Any, Dict, List, Optional

from src.rag.rag_engine import RAGEngine
from src.tool_use.tools import (
    stock_price,
    company_fundamentals,
    company_events,
    internet_search,
)


class MarketIntelligenceAgent:
    """
    Agente encargado de recopilar información objetiva del mercado.

    No recomienda. Solo reúne hechos:
    - precio
    - fundamentales
    - eventos
    - contexto externo
    - contexto recuperado por RAG
    """

    TICKER_BLACKLIST = {
        "RAG", "LLM", "API", "JSON", "USA", "ETF", "CEO", "CFO", "SEC",
        "NASDAQ", "NYSE", "USD", "EUR", "AI", "IPO", "Q1", "Q2", "Q3", "Q4"
    }

    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine

    def _safe_invoke(self, tool_obj: Any, arguments: Dict[str, Any]) -> str:
        try:
            return tool_obj.invoke(arguments)
        except Exception as e:
            return f"Error ejecutando herramienta: {str(e)}"

    def _is_error_response(self, value: Optional[str]) -> bool:
        if value is None:
            return True
        lower = value.lower()
        return lower.startswith("error") or "no se pudo" in lower or "no se encontró" in lower

    def _extract_ticker_from_text(self, text: str) -> Optional[str]:
        """
        Heurística para extraer tickers de resultados de búsqueda.
        """
        if not text:
            return None

        patterns = [
            r"\(([A-Z]{1,5})\)",                         # (NVDA)
            r"(?:NASDAQ|NYSE)\s*[:\-]?\s*([A-Z]{1,5})", # NASDAQ: NVDA
            r"\b([A-Z]{2,5})\b",                        # NVDA
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for candidate in matches:
                if candidate not in self.TICKER_BLACKLIST:
                    return candidate

        return None

    def _resolve_ticker(self, company_name: Optional[str]) -> Optional[str]:
        """
        Si no tenemos ticker, intenta inferirlo mediante búsqueda.
        """
        if not company_name:
            return None

        query = f"{company_name} stock ticker"
        search_result = self._safe_invoke(internet_search, {"query": query})

        if self._is_error_response(search_result):
            return None

        return self._extract_ticker_from_text(search_result)

    def _build_search_query(self, company_name: Optional[str], ticker: Optional[str]) -> Optional[str]:
        if company_name and ticker:
            return f"{company_name} {ticker} latest company news market outlook"
        if ticker:
            return f"{ticker} latest company news market outlook"
        if company_name:
            return f"{company_name} latest company news market outlook"
        return None

    def _build_rag_query(self, company_name: Optional[str], ticker: Optional[str]) -> Optional[str]:
        if company_name and ticker:
            return f"{company_name} {ticker} economía finanzas riesgos crecimiento valoración"
        if ticker:
            return f"{ticker} economía finanzas riesgos crecimiento valoración"
        if company_name:
            return f"{company_name} economía finanzas riesgos crecimiento valoración"
        return None

    def _has_useful_market_evidence(
        self,
        price_data: Optional[str],
        fundamentals_data: Optional[str],
        events_data: Optional[str],
        external_context: Optional[str],
        rag_snippets: List[Dict[str, Any]],
    ) -> bool:
        signals = 0

        if price_data and not self._is_error_response(price_data):
            signals += 1
        if fundamentals_data and not self._is_error_response(fundamentals_data):
            signals += 1
        if events_data and not self._is_error_response(events_data):
            signals += 1
        if external_context and not self._is_error_response(external_context):
            signals += 1
        if rag_snippets:
            signals += 1

        return signals >= 2

    def run(
        self,
        company_name: Optional[str],
        ticker: Optional[str],
        top_k_rag: int = 3,
    ) -> Dict[str, Any]:
        # 1. Validación inicial
        if not company_name and not ticker:
            return {
                "company_name": None,
                "ticker": None,
                "price_data": None,
                "fundamentals_data": None,
                "events_data": None,
                "external_context": None,
                "rag_context": [],
                "summary": "No se pudo identificar la empresa ni el ticker.",
                "resolved_ticker": False,
                "has_minimum_evidence": False,
                "error": "No se pudo identificar la empresa o ticker.",
            }

        # 2. Si falta ticker, intentamos resolverlo
        resolved_ticker = False
        if not ticker and company_name:
            resolved = self._resolve_ticker(company_name)
            if resolved:
                ticker = resolved
                resolved_ticker = True

        price_data = None
        fundamentals_data = None
        events_data = None

        # 3. Solo llamamos tools de mercado si tenemos ticker
        if ticker:
            price_data = self._safe_invoke(stock_price, {"ticker": ticker})
            fundamentals_data = self._safe_invoke(company_fundamentals, {"ticker": ticker})
            events_data = self._safe_invoke(company_events, {"ticker": ticker})

        # 4. Búsqueda externa solo si conocemos empresa/ticker
        external_context = None
        search_query = self._build_search_query(company_name, ticker)
        if search_query:
            external_context = self._safe_invoke(internet_search, {"query": search_query})

        # 5. RAG solo si conocemos empresa/ticker
        rag_snippets: List[Dict[str, Any]] = []
        rag_query = self._build_rag_query(company_name, ticker)
        if rag_query:
            rag_context = self.rag_engine.retrieve_context(
                query=rag_query,
                top_k=top_k_rag,
                similarity_threshold=0.75,
            )
            rag_snippets = [
                {
                    "label": ctx["label"],
                    "distance": ctx["distance"],
                    "text": ctx["text"][:800],
                }
                for ctx in rag_context
            ]

        has_minimum_evidence = self._has_useful_market_evidence(
            price_data=price_data,
            fundamentals_data=fundamentals_data,
            events_data=events_data,
            external_context=external_context,
            rag_snippets=rag_snippets,
        )

        # 6. Resumen
        summary_parts = []
        if company_name:
            summary_parts.append(f"Empresa analizada: {company_name}.")
        if ticker:
            summary_parts.append(f"Ticker analizado: {ticker}.")
        if resolved_ticker:
            summary_parts.append("El ticker se resolvió automáticamente a partir del nombre de la empresa.")
        if price_data and not self._is_error_response(price_data):
            summary_parts.append("Se ha obtenido información de precio.")
        if fundamentals_data and not self._is_error_response(fundamentals_data):
            summary_parts.append("Se han obtenido fundamentales.")
        if events_data and not self._is_error_response(events_data):
            summary_parts.append("Se han recuperado eventos recientes.")
        if external_context and not self._is_error_response(external_context):
            summary_parts.append("Se ha realizado búsqueda externa.")
        if rag_snippets:
            summary_parts.append(f"Se han recuperado {len(rag_snippets)} fragmentos por RAG.")
        if not has_minimum_evidence:
            summary_parts.append("La evidencia recuperada es limitada para emitir una recomendación sólida.")

        result = {
            "company_name": company_name,
            "ticker": ticker,
            "price_data": price_data,
            "fundamentals_data": fundamentals_data,
            "events_data": events_data,
            "external_context": external_context,
            "rag_context": rag_snippets,
            "summary": " ".join(summary_parts).strip(),
            "resolved_ticker": resolved_ticker,
            "has_minimum_evidence": has_minimum_evidence,
        }

        if not has_minimum_evidence:
            result["error"] = "No hay suficiente evidencia de mercado para emitir una recomendación fiable."

        return result