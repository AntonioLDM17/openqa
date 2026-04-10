import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Cambiad este nombre si finalmente usamos otro checkpoint
FIN_MODEL_NAME = "SUFE-AIFLM-Lab/Fin-R1"


def load_fin_model(model_name: str = FIN_MODEL_NAME):
    """
    Carga el modelo financiero Fin-R1 y su tokenizer.

    Args:
        model_name: nombre del modelo en Hugging Face o ruta local.

    Returns:
        Tuple (model, tokenizer)
    """
    print(f"Cargando modelo financiero desde {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    return model, tokenizer


def _build_prompt(prompt: str) -> str:
    """
    Construye un prompt simple y consistente.
    """
    return f"USER: {prompt}\nASSISTANT:"


def generate_financial_reasoning(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> str:
    """
    Genera una respuesta usando Fin-R1.

    Args:
        prompt: texto de entrada
        model: modelo cargado
        tokenizer: tokenizer cargado
        max_new_tokens: máximo de tokens a generar
        do_sample: si True, activa sampling
        temperature: temperatura de generación

    Returns:
        Texto generado
    """
    text = _build_prompt(prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }

    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    model, tokenizer = load_fin_model()

    test_prompt = (
        "Analiza si Nvidia podría ser interesante para un inversor moderado "
        "a 12 meses, teniendo en cuenta crecimiento, valoración y riesgos."
    )

    result = generate_financial_reasoning(
        test_prompt,
        model,
        tokenizer,
        max_new_tokens=300,
    )
    print(result)