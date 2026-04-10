from src.rlm.inference import load_rlm_model, generate_reasoning


def load_general_model():
    """
    Wrapper para cargar el modelo general del proyecto.

    Returns:
        Tuple (model, tokenizer)
    """
    return load_rlm_model()


def generate_general_reasoning(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
):
    """
    Wrapper para mantener una interfaz homogénea con fin_model.py.

    Args:
        prompt: texto de entrada
        model: modelo general cargado
        tokenizer: tokenizer del modelo
        max_new_tokens: mantenido por compatibilidad de interfaz

    Returns:
        Texto generado
    """
    # De momento reutilizamos directamente la función existente.
    # max_new_tokens no se usa aquí porque generate_reasoning ya
    # encapsula la generación interna en nuestro proyecto actual.
    return generate_reasoning(prompt, model, tokenizer)


if __name__ == "__main__":
    model, tokenizer = load_general_model()
    test_prompt = "Resume por qué una empresa con buenos fundamentales puede no ser una buena inversión a corto plazo."
    result = generate_general_reasoning(test_prompt, model, tokenizer)
    print(result)