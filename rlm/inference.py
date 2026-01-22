import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ruta a tu modelo final de fase 1
MODEL_PATH = "./weights/sft_lora_gsm8k"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

def load_rlm_model():
    # TODO: Cargar el modelo base y el adaptador LoRA
    print(f"Cargando modelo RLM desde {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    return model, tokenizer

def generate_reasoning(prompt, model, tokenizer):
    """
    Genera una respuesta que incluye el razonamiento (CoT).
    """
    # TODO: Implementar la generación
    text = f"""
    USER: {prompt}
    ASSISTANT:"""
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**input_ids, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0])
    return response

if __name__ == "__main__":
    # Prueba local
    model, tokenizer = load_rlm_model()
    test_prompt = "It's Ava's birthday party. Her parents bought a unicorn piñata for $13 and filled it with all of her favorite treats. They bought 4 bags of Reese's for $9 per bag, 3 bags of Snickers for $5 per bag, and 5 bags of Skittles for $7 per bag. How much did the unicorn piñata and the treats cost altogether?"
    print(generate_reasoning(test_prompt, model, tokenizer))
