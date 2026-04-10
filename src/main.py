"""
Main execution loop del sistema multiagente de inversión.

Uso:
    python -m src.main
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.general_model import load_general_model
from src.models.fin_model import load_fin_model
from src.rag.rag_engine import RAGEngine
from src.agents.investment_multiagent_system import InvestmentMultiAgentSystem


def main():
    print("=" * 60)
    print("SISTEMA MULTIAGENTE DE RECOMENDACIÓN DE INVERSIONES")
    print("=" * 60)

    # 1. Cargar modelo general
    print("\nCargando modelo general...")
    general_model, general_tokenizer = load_general_model()

    # 2. Cargar modelo financiero
    print("\nCargando modelo financiero Fin-R1...")
    fin_model, fin_tokenizer = load_fin_model()

    # 3. Inicializar motor RAG
    print("\nInicializando motor RAG...")
    rag_engine = None
    try:
        rag_engine = RAGEngine()
        print("Motor RAG inicializado correctamente.")
    except Exception as e:
        print(f"RAG no disponible: {e}")
        print("Ejecuta 'python -m src.rag.load_dataset' si necesitas cargar los datos.")
        return

    # 4. Crear sistema multiagente
    print("\nMontando sistema multiagente...")
    system = InvestmentMultiAgentSystem(
        general_model=general_model,
        general_tokenizer=general_tokenizer,
        fin_model=fin_model,
        fin_tokenizer=fin_tokenizer,
        rag_engine=rag_engine,
    )

    # 5. Preguntas de prueba
    questions = [
        "Analiza Nvidia y dime si tendría sentido entrar ahora para un inversor moderado a 12 meses.",
        # "Compara Microsoft y Amazon para un perfil conservador a medio plazo.",
        # "Analiza Tesla y dime si el riesgo actual compensa para un perfil agresivo.",
    ]

    for question in questions:
        print(f"\n{'#' * 60}")
        print(f"PREGUNTA: {question}")
        print(f"{'#' * 60}")

        result = system.run(question)

        print("\nRESPUESTA FINAL:")
        print(result["final_answer"])

        print("\nTRAZA:")
        for step in result.get("trace", []):
            print(f"- Step {step.get('step')} | Agent: {step.get('agent')}")

        print("=" * 60)


if __name__ == "__main__":
    main()