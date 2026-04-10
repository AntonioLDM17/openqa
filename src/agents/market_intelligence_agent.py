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

    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine

    def _safe_invoke(self, tool_obj: Any, arguments: Dict[str, Any]) -> str:
        try:
            return tool_obj.invoke(arguments)
        except Exception as e:
            return f"Error ejecutando herramienta: {str(e)}"

    def _build_search_query(self, company_name: Optional[str], ticker: Optional[str]) -> str:
        if company_name and ticker:
            return f"{company_name} {ticker} latest company news market outlook"
        if ticker:
            return f"{ticker} latest company news market outlook"
        if company_name:
            return f"{company_name} latest company news market outlook"
        return "latest company news market outlook"

    def _build_rag_query(self, company_name: Optional[str], ticker: Optional[str]) -> str:
        if company_name and ticker:
            return f"{company_name} {ticker} economía finanzas riesgos crecimiento valoración"
        if ticker:
            return f"{ticker} economía finanzas riesgos crecimiento valoración"
        if company_name:
            return f"{company_name} economía finanzas riesgos crecimiento valoración"
        return "finanzas valoración riesgos crecimiento"

    def run(
        self,
        company_name: Optional[str],
        ticker: Optional[str],
        top_k_rag: int = 3,
    ) -> Dict[str, Any]:
        price_data = None
        fundamentals_data = None
        events_data = None

        if ticker:
            price_data = self._safe_invoke(stock_price, {"ticker": ticker})
            fundamentals_data = self._safe_invoke(company_fundamentals, {"ticker": ticker})
            events_data = self._safe_invoke(company_events, {"ticker": ticker})

        search_query = self._build_search_query(company_name, ticker)
        external_context = self._safe_invoke(internet_search, {"query": search_query})

        rag_query = self._build_rag_query(company_name, ticker)
        rag_context = self.rag_engine.retrieve_context(
            query=rag_query,
            top_k=top_k_rag,
            similarity_threshold=0.75,
        )

        rag_snippets: List[Dict[str, Any]] = [
            {
                "label": ctx["label"],
                "distance": ctx["distance"],
                "text": ctx["text"][:800],
            }
            for ctx in rag_context
        ]

        summary_parts = []
        if ticker:
            summary_parts.append(f"Ticker analizado: {ticker}.")
        if company_name:
            summary_parts.append(f"Empresa analizada: {company_name}.")
        if price_data:
            summary_parts.append("Se ha obtenido información de precio.")
        if fundamentals_data:
            summary_parts.append("Se han obtenido fundamentales.")
        if events_data:
            summary_parts.append("Se han recuperado eventos recientes.")
        if external_context:
            summary_parts.append("Se ha realizado búsqueda externa.")
        if rag_snippets:
            summary_parts.append(f"Se han recuperado {len(rag_snippets)} fragmentos por RAG.")

        return {
            "company_name": company_name,
            "ticker": ticker,
            "price_data": price_data,
            "fundamentals_data": fundamentals_data,
            "events_data": events_data,
            "external_context": external_context,
            "rag_context": rag_snippets,
            "summary": " ".join(summary_parts).strip(),
        }