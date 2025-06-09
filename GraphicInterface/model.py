import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sentence_transformers import SentenceTransformer

# === Конфигурация ===
MERGED_MODEL_PATH = "./merged_model"  # Путь к объединённой модели
EXAMPLES_STORE_PATH = "./examples_store.json"

# Загружаем объединённую модель и токенизатор
tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Загружаем модель для эмбеддингов
model_embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# === Загрузка примеров из файла ===
def load_examples_store(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_similar_examples(input_text, examples_store, top_k=3):
    input_embedding = model_embeddings.encode(input_text)
    similarities = []
    
    for category, data in examples_store.items():
        for example in data['examples']:
            similarity = cosine_similarity(input_embedding, example['embedding'])
            similarities.append((similarity, example))
    
    # Сортируем по убыванию косинусной близости
    similarities.sort(reverse=True)
    
    # Возвращаем top_k наиболее похожих примеров
    return [example for _, example in similarities[:top_k]]

def generate_examples(input_text, examples_store_path='examples_store.json'):
    examples_store = load_examples_store(examples_store_path)
    similar_examples = get_similar_examples(input_text, examples_store)
    
    # Здесь можно добавить логику для генерации новых примеров на основе similar_examples
    return similar_examples

# === Сохранение примеров в файл ===
def save_examples_store(store):
    with open(EXAMPLES_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


# === Подсчет числа характеристик в output ===
def count_fields(output_str):
    return len([p for p in output_str.split("/sprt/") if ":" in p and p.split(":")[1].strip() != ""])


# === Обновление пула примеров для категории ===
def update_example_store(store, category, input_text, output_text):
    if category not in store:
        store[category] = {"examples": []}

    # Вычисляем эмбеддинг для нового примера
    input_embedding = model_embeddings.encode(input_text)

    # Проверяем, достаточно ли отличается новый пример от существующих
    is_different = True
    for ex in store[category]["examples"]:
        similarity = cosine_similarity(input_embedding, ex["embedding"])
        if similarity > 0.7:  # Порог косинусной близости
            is_different = False
            break

    if is_different:
        store[category]["examples"].append({
            "input": input_text,
            "output": output_text,
            "embedding": input_embedding.tolist()
        })


# === Формирование few-shot подсказки по категории ===
def build_few_shot_prompt(category, store, input_text):
    prompt = ""
    if category in store:
        examples = []
        if "full" in store[category]:
            examples.append(store[category]["full"])
        if "partial" in store[category]:
            examples.append(store[category]["partial"])

        for ex in examples:
            prompt += f"INPUT: {ex['input']}\nOUTPUT: {ex['output']}{tokenizer.eos_token}\n\n"

    prompt += f"INPUT: {input_text}\nOUTPUT:"
    return prompt


# === Генерация с few-shot и автоматическим пополнением ===
def generate_with_few_shot_dynamic(input_text):
    store = load_examples_store(EXAMPLES_STORE_PATH)

    # Получаем три наиболее похожих примера
    similar_examples = get_similar_examples(input_text, store)

    # Формируем подсказку с использованием похожих примеров
    prompt = ""
    for ex in similar_examples:
        prompt += f"INPUT: {ex['input']}\nOUTPUT: {ex['output']}{tokenizer.eos_token}\n\n"

    prompt += f"INPUT: {input_text}\nOUTPUT:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

    output_start = decoded.find("OUTPUT:")
    if output_start != -1:
        final_output = decoded[output_start + len("OUTPUT:"):].strip()
    else:
        final_output = decoded.strip()

    # Удалим </s> в конце, если есть
    if final_output.endswith("</s>"):
        final_output = final_output[:-len("</s>")].strip()

    # Определяем категорию
    category = "не определено"
    for part in final_output.split("/sprt/"):
        if part.startswith("Категория:"):
            val = part.split(":", 1)[1].strip()
            if val:
                category = val
            break

    # Обновление примеров
    update_example_store(store, category, input_text, final_output)
    save_examples_store(store)
    return final_output

# Пример использования
if __name__ == "__main__":
    input_text = "Пример входного текста"
    similar_examples = generate_examples(input_text)
    print("Наиболее похожие примеры:", similar_examples)
