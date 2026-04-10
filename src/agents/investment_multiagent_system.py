from typing import Any, Dict, Optional

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

        final_answer = critic_report.get("final_answer") or recommendation_report.get("thesis", "")

        return {
            "final_answer": final_answer,
            "orchestration": orchestration,
            "market_report": market_report,
            "recommendation_report": recommendation_report,
            "critic_report": critic_report,
            "trace": trace,
        }