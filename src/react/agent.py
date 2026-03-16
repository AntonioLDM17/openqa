import json
# Importar componentes de fases anteriores
# from phase1_training.inference import generate_reasoning, load_rlm_model
# from phase2_tool_use.tools import get_tools_prompt
# from phase2_tool_use.tool_handler import parse_and_execute_tool_call
# from phase3_rag.rag_engine import retrieve_context

class ReActAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tools_prompt = "TODO: Obtener prompt de herramientas (Fase 2)"
        # MEJORAR EL PROMPT DEL SISTEMA. Mirar langsmith, por ejemplo: https://smith.langchain.com/hub/hwchase17
        self.system_prompt = f"""Eres un agente autónomo útil.
Usa el siguiente formato de Pensamiento-Acción-Observación para resolver tareas complejas.

{self.tools_prompt}

Formato a seguir:
Thought: Debo pensar qué hacer a continuación.
Action: 
Observation: El resultado de la herramienta.
... (repetir hasta tener la respuesta final)
Thought: Ya tengo suficiente información.
Final Answer: La respuesta final al usuario.


"""

    def run(self, user_query, max_steps=5):
        """
        Ejecuta el bucle ReAct para resolver la query.
        """
        history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        trace = [] # Para guardar los pasos dados y mostrarlos en la API

        step = 0
        while step < max_steps:
            # 1. Generar pensamiento (Thought) y posible Acción
            # TODO: Llamar al modelo de Fase 1 con el historial actual
            # model_output = generate_reasoning(current_prompt, self.model, self.tokenizer)
            
            print(f"--- Step {step} ---")
            # print(f"Model Output: {model_output}")
            model_output = "Placeholder del modelo (Thought + Action placeholder)" # TODO Remove
            
            # Añadir output al historial y al trace
            history.append({"role": "assistant", "content": model_output})
            trace.append({"step": step, "type": "model_output", "content": model_output})

            # 2. Detectar si hay "Final Answer"
            # TODO: Implementar la detección de "Final Answer" esto debe ser modificado según las
            # necesitadades de cada equipo, en el caso más sencillo se puede buscar el string "Final Answer:" en el modelo output. 
            # En casos más complejos se puede usar un regex o un modelo de clasificación (con LLM incluso) para detectar si la respuesta final satisface la pregunta del usuario.
            if "Final Answer:" in model_output:
                # Extraer respuesta final
                final_answer = model_output.split("Final Answer:")[-1].strip()
                return {"final_answer": final_answer, "trace": trace}

            # 3. Intentar ejecutar Acción (Herramienta)
            # TODO: Usar el handler de Fase 2 y RAG de Fase 3 para ver si hay JSON de herramienta
            # tool_result = parse_and_execute_tool_call(model_output)
            tool_result = None # Placeholder

            if tool_result:
                print(f"Observation: {tool_result}")
                observation_msg = f"Observation: {tool_result}"
                history.append({"role": "user", "content": observation_msg}) # Se suele añadir como rol user o system
                trace.append({"step": step, "type": "observation", "content": tool_result})
            else:
                # Si no hubo herramienta ni respuesta final, forzar al modelo a continuar o parar.
                if "Action:" in model_output and not tool_result:
                    history.append({"role": "user", "content": "Observation: Error: No se pudo ejecutar la acción. Revisa el formato JSON."})
                else:
                    # El modelo solo pensó, dejar que siga en el siguiente loop
                    pass
            
            step += 1

        return {"final_answer": "Error: Se excedió el número máximo de pasos.", "trace": trace}

# Ejemplo de uso (si se ejecuta directamente)
if __name__ == "__main__":
    model, tokenizer = None, None 
    agent = ReActAgent(model, tokenizer)
    # response = agent.run("¿Cuál es la raíz cuadrada de la edad del presidente de Francia?")
    # print(response)
