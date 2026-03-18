from rlm.inference import load_rlm_model
from tool_use.tool_handler import run_agent_loop, get_model

model_path = "./rlm/weights/sft_lora_gsm8k"
base_model = "Qwen/Qwen2.5-7B-Instruct"

def main(questions: list[str]):

    # model = load_rlm_model(base_model, model_path)
    model = get_model()

    for q in questions:
        run_agent_loop(model, q, max_iterations=5, verbose=True)
        print("====")

if __name__ == "__main__":

    questions = [
        "¿Cuánto es 15 * 3 + 50?",
        "Cuando saldrá el próximo Call of Duty?",
        "Dame un resumen de la situación financiera básica de Microsoft.",
        "¿Ha pasado algo relevante recientemente con Tesla que deba preocupar a los inversores?",
        "¿Cómo se está comportando la acción de NVIDIA hoy en el mercado?"
    ]

    main(questions)