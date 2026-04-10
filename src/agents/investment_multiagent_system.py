from typing import Any, Dict

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.market_intelligence_agent import MarketIntelligenceAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.critic_risk_agent import CriticRiskAgent
from src.rag.rag_engine import RAGEngine


class InvestmentMultiAgentSystem:
    """
    Sistema multiagente completo para recomendación de inversiones.

    Flujo:
    1. Orchestrator
    2. Market Intelligence
    3. Recommendation (Fin-R1)
    4. Critic / Risk
    5. Respuesta final + trace
    """

    def __init__(
        self,
        general_model: Any,
        general_tokenizer: Any,
        fin_model: Any,
        fin_tokenizer: Any,
        rag_engine: RAGEngine,
    ):
        self.orchestrator = OrchestratorAgent(general_model, general_tokenizer)
        self.market_agent = MarketIntelligenceAgent(rag_engine)
        self.recommendation_agent = RecommendationAgent(fin_model, fin_tokenizer)
        self.critic_agent = CriticRiskAgent(general_model, general_tokenizer)

    def _build_failure_response(
        self,
        message: str,
        trace: list,
        orchestration: Dict[str, Any] | None = None,
        market_report: Dict[str, Any] | None = None,
        recommendation_report: Dict[str, Any] | None = None,
        critic_report: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return {
            "final_answer": message,
            "orchestration": orchestration or {},
            "market_report": market_report or {},
            "recommendation_report": recommendation_report or {},
            "critic_report": critic_report or {},
            "trace": trace,
        }

    def _recommendation_is_too_weak(self, recommendation_report: Dict[str, Any]) -> bool:
        thesis = recommendation_report.get("thesis", "")
        strengths = recommendation_report.get("strengths", [])
        risks = recommendation_report.get("risks", [])
        scenarios = recommendation_report.get("scenarios", [])

        if not thesis or len(thesis.strip()) < 20:
            return True

        if not isinstance(strengths, list):
            strengths = []
        if not isinstance(risks, list):
            risks = []
        if not isinstance(scenarios, list):
            scenarios = []

        # Si no devuelve casi nada estructurado, la consideramos demasiado débil
        if len(strengths) == 0 and len(risks) == 0 and len(scenarios) == 0:
            return True

        return False

    def run(self, user_query: str) -> Dict[str, Any]:
        trace = []

        # 1. Orchestrator
        orchestration = self.orchestrator.run(user_query)
        trace.append({
            "step": 1,
            "agent": "orchestrator",
            "output": orchestration,
        })

        company_name = orchestration.get("company_name")
        ticker = orchestration.get("ticker")
        risk_profile = orchestration.get("risk_profile", "moderado")
        horizon = orchestration.get("horizon", "12 meses")

        # Si el orquestador no ha detectado ni ticker ni empresa, abortamos pronto
        if not company_name and not ticker:
            return self._build_failure_response(
                message=(
                    "No he podido identificar con suficiente claridad la empresa o el ticker "
                    "sobre el que quieres análisis. Indica el nombre de la empresa o su ticker bursátil."
                ),
                trace=trace,
                orchestration=orchestration,
            )

        # 2. Market Intelligence
        market_report = self.market_agent.run(
            company_name=company_name,
            ticker=ticker,
        )
        trace.append({
            "step": 2,
            "agent": "market_intelligence",
            "output": market_report,
        })

        # Si el market agent no consigue evidencia suficiente, no seguimos
        if market_report.get("error") or not market_report.get("has_minimum_evidence", False):
            return self._build_failure_response(
                message=(
                    "No he podido reunir suficiente evidencia estructurada y fiable del mercado "
                    "para emitir una recomendación razonada sobre esta empresa en este momento."
                ),
                trace=trace,
                orchestration=orchestration,
                market_report=market_report,
            )

        # 3. Recommendation Agent (Fin-R1)
        recommendation_report = self.recommendation_agent.run(
            market_report=market_report,
            risk_profile=risk_profile,
            horizon=horizon,
            user_query=user_query,
        )
        trace.append({
            "step": 3,
            "agent": "recommendation",
            "output": recommendation_report,
        })

        # Si recommendation ha producido una salida muy pobre, devolvemos respuesta prudente
        if self._recommendation_is_too_weak(recommendation_report):
            return self._build_failure_response(
                message=(
                    "He podido recuperar información de mercado, pero la tesis de inversión generada "
                    "no tiene suficiente calidad o detalle como para devolver una recomendación fiable."
                ),
                trace=trace,
                orchestration=orchestration,
                market_report=market_report,
                recommendation_report=recommendation_report,
            )

        # 4. Critic / Risk Agent
        critic_report = self.critic_agent.run(
            market_report=market_report,
            recommendation_report=recommendation_report,
            risk_profile=risk_profile,
            horizon=horizon,
            user_query=user_query,
        )
        trace.append({
            "step": 4,
            "agent": "critic_risk",
            "output": critic_report,
        })

        # Si el critic dice que no hay evidencia suficiente, prevalece esa evaluación
        if not critic_report.get("enough_evidence", False):
            final_answer = (
                critic_report.get("final_answer")
                or (
                    "No hay suficiente evidencia para emitir una recomendación de inversión "
                    "con un nivel razonable de confianza."
                )
            )
            return {
                "final_answer": final_answer,
                "orchestration": orchestration,
                "market_report": market_report,
                "recommendation_report": recommendation_report,
                "critic_report": critic_report,
                "trace": trace,
            }

        # Si hay evidencia suficiente pero el critic detecta falta de grounding,
        # seguimos usando su respuesta final porque es el agente verificador.
        final_answer = (
            critic_report.get("final_answer")
            or recommendation_report.get("thesis")
            or "No se pudo generar una respuesta final."
        )

        return {
            "final_answer": final_answer,
            "orchestration": orchestration,
            "market_report": market_report,
            "recommendation_report": recommendation_report,
            "critic_report": critic_report,
            "trace": trace,
        }