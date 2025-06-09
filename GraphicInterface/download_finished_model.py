import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

# === Конфигурация ===
BASE_MODEL = "TheBloke/Llama-2-7B-Chat-AWQ"
LOCAL_MODEL_PATH = "./llama_model"
LORA_ADAPTER_PATH = "./adapter"
MERGED_MODEL_PATH = "./merged_model"

def download_and_merge_model():
    # === Шаг 1: Скачивание модели ===
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Скачиваем квантованную AWQ модель...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        print("Сохраняем модель локально...")
        model.save_pretrained(LOCAL_MODEL_PATH)
        tokenizer.save_pretrained(LOCAL_MODEL_PATH)
    else:
        print(f"Модель уже существует в {LOCAL_MODEL_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

    # === Шаг 2: Наложение LoRA ===
    print("Накладываем LoRA адаптер...")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)

    # === Шаг 3: Сохраняем объединённую модель ===
    print(f"Сохраняем объединённую модель в {MERGED_MODEL_PATH}...")
    model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    print("Готово!")

if __name__ == "__main__":
    download_and_merge_model()
