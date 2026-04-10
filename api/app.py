from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sys
import os

# Añadir el directorio raíz al path para poder importar los módulos del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- IMPORTACIONES DEL PROYECTO BASE ---
from src.models.general_model import load_general_model, generate_general_reasoning
from src.models.fin_model import load_fin_model
from src.tool_use.tool_handler import run_agent_loop
from src.rag.rag_engine import RAGEngine
from src.react.agent import ReActAgent

# --- IMPORTACIONES DEL SISTEMA MULTIAGENTE ---
from src.agents.investment_multiagent_system import InvestmentMultiAgentSystem

app = FastAPI(
    title="Práctica Master: Modelos Generativos Profundos",
    description="API para evaluar las 4 fases de la práctica y el sistema multiagente de inversión."
)

# --- Variables Globales (Modelos) ---
GENERAL_MODEL = None
GENERAL_TOKENIZER = None
FIN_MODEL = None
FIN_TOKENIZER = None
AGENT = None
RAG_ENGINE = None
INVESTMENT_SYSTEM = None


@app.on_event("startup")
async def startup_event():
    global GENERAL_MODEL, GENERAL_TOKENIZER
    global FIN_MODEL, FIN_TOKENIZER
    global AGENT, RAG_ENGINE, INVESTMENT_SYSTEM

    print("Inicializando API...")

    # Modelo general
    GENERAL_MODEL, GENERAL_TOKENIZER = load_general_model()

    # Modelo financiero
    FIN_MODEL, FIN_TOKENIZER = load_fin_model()

    # Motor RAG
    RAG_ENGINE = RAGEngine()

    # Agente ReAct antiguo (fase 4)
    if GENERAL_MODEL and GENERAL_TOKENIZER:
        AGENT = ReActAgent(GENERAL_MODEL, GENERAL_TOKENIZER)

    # Sistema multiagente nuevo
    if (
        GENERAL_MODEL and GENERAL_TOKENIZER and
        FIN_MODEL and FIN_TOKENIZER and
        RAG_ENGINE
    ):
        INVESTMENT_SYSTEM = InvestmentMultiAgentSystem(
            general_model=GENERAL_MODEL,
            general_tokenizer=GENERAL_TOKENIZER,
            fin_model=FIN_MODEL,
            fin_tokenizer=FIN_TOKENIZER,
            rag_engine=RAG_ENGINE,
        )

    print("Modelos cargados correctamente.")


# --- Modelos de Pydantic para Request/Response ---
class QueryRequest(BaseModel):
    prompt: str


class GenericResponse(BaseModel):
    response: str
    trace: list[dict] = []
    details: dict = {}


# ================= ENDPOINTS DE EVALUACIÓN =================

# --- FASE 1: Razonamiento (RLM) ---
@app.post("/phase1/reasoning", response_model=GenericResponse, tags=["Fase 1"])
async def phase1_endpoint(request: QueryRequest):
    """
    Evalúa el modelo RLM. Debe devolver la respuesta con el razonamiento (CoT) visible.
    """
    if not GENERAL_MODEL or not GENERAL_TOKENIZER:
        return {
            "response": "ERROR: Modelo de Fase 1 no cargado.",
            "details": {"status": "todo"},
        }

    response_text = generate_general_reasoning(
        request.prompt,
        GENERAL_MODEL,
        GENERAL_TOKENIZER,
    )
    print("Response Text:", response_text)

    try:
        reasoning, response = response_text.split("ASSISTANT:")[1].split("Final answer:")
    except Exception:
        reasoning, response = response_text, response_text

    return {
        "response": response,
        "trace": [{"step": 0, "content": reasoning}],
        "details": {"stage": "sft_grpo"},
    }


# --- FASE 2: Tool Use ---
@app.post("/phase2/tools", response_model=GenericResponse, tags=["Fase 2"])
async def phase2_endpoint(request: QueryRequest):
    """
    Evalúa la capacidad de llamar herramientas.
    Si el prompt requiere una herramienta, debe devolver la ejecución simulada.
    """
    if not GENERAL_MODEL or not GENERAL_TOKENIZER:
        return {
            "response": "ERROR: Modelo no cargado.",
            "details": {"status": "error"},
        }

    tool_result = run_agent_loop(GENERAL_MODEL, request.prompt, GENERAL_TOKENIZER)

    if tool_result:
        return {
            "response": f"Tool execution result: {tool_result[-1]['content'].replace('</s>', '')}",
            "details": {"tool_called": True},
            "trace": tool_result[1:],
        }

    return {
        "response": "No tool call detected or needed.",
        "details": {"tool_called": False},
    }


# --- FASE 3: RAG ---
@app.post("/phase3/rag", response_model=GenericResponse, tags=["Fase 3"])
async def phase3_endpoint(request: QueryRequest):
    """
    Evalúa el RAG. Recupera contexto de ChromaDB y genera una respuesta.
    """
    if not RAG_ENGINE:
        return {
            "response": "ERROR: Motor RAG no inicializado.",
            "details": {"status": "error"},
        }

    # 1. Recuperar contexto
    context_list = RAG_ENGINE.retrieve_context(
        request.prompt,
        top_k=5,
        similarity_threshold=0.75
    )

    # 2. Formatear prompt con el contexto recuperado
    rag_prompt = RAG_ENGINE.format_rag_prompt(request.prompt, context_list)

    # 3. Generar respuesta con el modelo general
    if GENERAL_MODEL and GENERAL_TOKENIZER:
        response_text = generate_general_reasoning(
            rag_prompt,
            GENERAL_MODEL,
            GENERAL_TOKENIZER,
        )
        try:
            response_text = response_text.split("ASSISTANT:")[-1].strip()
        except Exception:
            pass
    else:
        response_text = "Modelo no disponible. Contexto recuperado correctamente."

    retrieved_docs = [
        {
            "text": ctx["text"][:200] + "...",
            "label": ctx["label"],
            "distance": ctx["distance"]
        }
        for ctx in context_list
    ]

    return {
        "response": response_text,
        "trace": [
            {"step": i, "content": f"[{ctx['label']}] {ctx['text'][:150]}..."}
            for i, ctx in enumerate(context_list)
        ],
        "details": {
            "retrieved_docs": retrieved_docs,
            "num_results": len(context_list),
        },
    }


# --- FASE 4: Agente ReAct ---
@app.post("/phase4/agent", tags=["Fase 4"])
async def phase4_endpoint(request: QueryRequest):
    """
    Evalúa el agente completo. Devuelve la respuesta final y la traza de ejecución.
    """
    if not AGENT:
        return {
            "final_answer": "ERROR: Agente no inicializado.",
            "trace": [],
        }

    result = AGENT.run(request.prompt)
    return result


# --- PROYECTO FINAL: Sistema multiagente de inversión ---
@app.post("/investment/recommendation", tags=["Proyecto Final"])
async def investment_recommendation_endpoint(request: QueryRequest):
    """
    Ejecuta el sistema multiagente de recomendación de inversiones.
    """
    if not INVESTMENT_SYSTEM:
        return {
            "final_answer": "ERROR: Sistema multiagente no inicializado.",
            "trace": [],
        }

    result = INVESTMENT_SYSTEM.run(request.prompt)
    return result


if __name__ == "__main__":
    # Para correr localmente: python api/app.py
    uvicorn.run(app, host="0.0.0.0", port=8045)