import json
import re
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from tools import calculator, simulated_search, internet_search

load_dotenv()

TOOL_SCHEMAS = {
    "calculator": {
        "name": "calculator",
        "description": "Evalúa una expresión matemática simple. Útil para realizar cálculos aritméticos.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "La expresión matemática a evaluar (ej: '25 * 4 + 100')"
                }
            },
            "required": ["expression"]
        }
    },
    # "simulated_search": {
    #     "name": "simulated_search",
    #     "description": "Busca información en una base de datos. SIEMPRE usa esta herramienta para buscar información sobre personas, lugares, tecnología o cualquier dato factual.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "query": {
    #                 "type": "string",
    #                 "description": "La consulta de búsqueda"
    #             }
    #         },
    #         "required": ["query"]
    #     }
    # },
    "internet_search": {
        "name": "internet_search",
        "description": "Busca información en internet usando Tavily API. Usa esta herramienta para información actualizada o no disponible en la base de datos.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La consulta de búsqueda en internet"
                }
            },
            "required": ["query"]
        }
    }
}

SYSTEM_PROMPT = """Eres un asistente inteligente que puede usar herramientas para responder preguntas.

Herramientas disponibles:
{tools_description}

INSTRUCCIONES IMPORTANTES:
Deberás decidir si quieres llamar a una herramienta. En tal caso, tu llamada deberá estar formateada de la siguiente forma:
{{"nombre": "nombre_de_la_herramienta", "argumentos": {{"parametro": "valor"}}}}

Tras la ejecución de la herramienta, recibirás el resultado de la llamada a la herramienta.
"""

def get_tools_description():
    """Genera una descripción legible de las herramientas disponibles."""
    descriptions = []
    for tool_name, schema in TOOL_SCHEMAS.items():
        params = ", ".join(schema["parameters"]["properties"].keys())
        descriptions.append(f"- {tool_name}({params}): {schema['description']}")
    return "\n".join(descriptions)

def parse_and_execute_tool_call(model_output):
    """
    Intenta detectar y ejecutar una llamada a herramienta en el output del modelo.
    Busca el formato JSON: {"nombre": "...", "argumentos": {...}}
    
    Retorna:
    - El resultado de la herramienta (str) si hubo una llamada exitosa.
    - None si no se detectó ninguna llamada válida.
    """
    
    available_tools = {
        "calculator": calculator,
        # "simulated_search": simulated_search,
        "internet_search": internet_search
    }
    
    try:
        # Buscar JSON con campos "nombre" y "argumentos"
        # Usar un patrón más robusto que maneje JSON anidado
        json_match = re.search(r'\{[^{}]*"nombre"[^{}]*"argumentos"[^{}]*\{[^{}]*\}[^{}]*\}', model_output, re.DOTALL)
        
        # Si no encuentra con argumentos anidados, buscar formato simple
        if not json_match:
            json_match = re.search(r'\{[^{}]*"nombre"[^{}]*"argumentos"[^{}]*\}', model_output, re.DOTALL)
        
        if not json_match:
            return None
        
        json_str = json_match.group(0).strip()
        tool_call = json.loads(json_str)
        
        # Usar los nombres en español: "nombre" y "argumentos"
        tool_name = tool_call.get("nombre")
        arguments = tool_call.get("argumentos", {})
        
        if not tool_name:
            return "Error: No se especificó el nombre de la herramienta."
        
        if tool_name not in available_tools:
            return f"Error: Herramienta '{tool_name}' no encontrada."
        
        tool_function = available_tools[tool_name]
        result = tool_function.invoke(arguments)
        
        return result
        
    except json.JSONDecodeError as e:
        return None
    except Exception as e:
        return f"Error ejecutando herramienta: {str(e)}"

def get_model():
    """Crea y retorna el modelo Ollama local."""
    return ChatOllama(
        model="llama3",
        temperature=0
    )

def run_agent_loop(user_question, max_iterations=5, verbose=True):
    """
    Ejecuta el loop ReAct: Model → Tool → Model → Answer
    
    Args:
        user_question: La pregunta del usuario
        max_iterations: Número máximo de iteraciones para evitar loops infinitos
        verbose: Si True, imprime el proceso paso a paso
    
    Returns:
        La respuesta final del modelo
    """
    model = get_model()
    
    system_prompt = SYSTEM_PROMPT.format(tools_description=get_tools_description())
    
    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Iteración {iteration + 1}")
            print(f"{'='*60}")
        
        response = model.invoke(conversation_history)
        model_output = response.content
        
        if verbose:
            print(f"\n🤖 Modelo dice:\n{model_output}")
        
        tool_result = parse_and_execute_tool_call(model_output)
        
        if tool_result is None:
            if verbose:
                print(f"\n✅ Respuesta final (sin herramienta)")
            return model_output
        
        if verbose:
            print(f"\n🔧 Resultado de herramienta:\n{tool_result}")
        
        conversation_history.append({"role": "assistant", "content": model_output})
        conversation_history.append({
            "role": "user", 
            "content": f"Resultado de la herramienta: {tool_result}\n\nAhora proporciona una respuesta final en lenguaje natural basándote en este resultado."
        })
    
    if verbose:
        print(f"\n⚠️ Alcanzado el máximo de iteraciones ({max_iterations})")
    
    final_response = model.invoke(conversation_history)
    return final_response.content

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PRUEBA DEL AGENTE ReAct")
    print("="*60)
    
    # Prueba simple con cálculo
    # question = "¿Cuánto es 15 * 3 + 50?"
    question = "Cuando saldrá el próximo Call of Duty?"
    print(f"\n❓ Pregunta: {question}\n")
    
    answer = run_agent_loop(question, verbose=True)
    
    print("\n" + "="*60)
    print("📝 RESPUESTA FINAL:")
    print("="*60)
    print(answer)
    print("\n")
