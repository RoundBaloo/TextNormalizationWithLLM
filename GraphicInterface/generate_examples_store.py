import openpyxl
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Загружаем модель для эмбеддингов
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def normalize_text(text):
    return ' '.join(text.lower().split())

def read_xlsx_to_nested_list(file_path):
    all_lists = []
    current_list = []
    headers_length = 0

    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active

    for row in sheet.iter_rows(values_only=True):
        if all(cell is None for cell in row):
            if current_list:
                all_lists.append(current_list)
                current_list = []
            continue

        if row[0] == "Полное название":
            current_list.append([cell for cell in row if cell is not None])
            headers_length = len(current_list[0])
        else:
            if current_list:
                current_row = []
                for i in range(headers_length):
                    if i < len(row) and row[i] is not None:
                        current_row.append(row[i])
                    else:
                        current_row.append('')
                current_list.append(current_row)

    if current_list:
        all_lists.append(current_list)

    all_lists = [lst for lst in all_lists if lst]

    return all_lists

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def create_examples_store(nested_lists):
    examples_store = {}
    
    for category_list in nested_lists:
        # Получаем заголовки
        headers = category_list[0]
        
        # Получаем название категории из первой строки данных
        if len(category_list) < 2:
            continue
            
        category_name = None
        for row in category_list[1:]:
            if row[1] and str(row[1]).strip():  # Ищем в столбце "Категория"
                category_name = str(row[1]).strip()
                break
                
        if not category_name:
            continue
            
        # Создаем список примеров для категории
        examples = []
        
        # Обрабатываем каждую строку данных
        for row in category_list[1:]:
            if not row[0] or not row[1]:  # Пропускаем строки без названия или категории
                continue
                
            input_text = normalize_text(str(row[0]))
            output_parts = [f"Категория:{category_name}"]
            
            # Добавляем характеристики, пропуская столбцы "Полное название" и "Категория"
            for i in range(2, len(headers)):
                if row[i] and str(row[i]).strip():
                    output_parts.append(f"{headers[i]}:{str(row[i]).strip()}")
            
            output_parts.append("Индекс заполненности:1")
            output_text = "/sprt/".join(output_parts)
            
            # Вычисляем эмбеддинг для входного текста
            embedding = model.encode(input_text).tolist()
            
            # Проверяем, достаточно ли отличается новый пример от существующих
            is_different = True
            for ex in examples:
                similarity = cosine_similarity(embedding, ex["embedding"])
                if similarity > 0.7:  # Порог косинусной близости
                    is_different = False
                    break
            
            if is_different:
                examples.append({
                    "input": input_text,
                    "output": output_text,
                    "embedding": embedding
                })
        
        # Добавляем категорию в общий словарь
        if examples:
            examples_store[category_name] = {"examples": examples}
    
    return examples_store

def main():
    file_path = r"D:\URFU\GUI\gui_ml\dataset_generated_new.xlsx"
    output_file = "examples_store.json"
    
    # Читаем данные из Excel
    print("Чтение Excel файла...")
    nested_lists = read_xlsx_to_nested_list(file_path)
    print(f"Прочитано категорий: {len(nested_lists)}")
    
    # Создаем examples_store
    print("Создание examples_store...")
    examples_store = create_examples_store(nested_lists)
    print(f"Создано категорий: {len(examples_store)}")
    print("Категории:", list(examples_store.keys()))
    
    # Сохраняем в JSON
    print(f"Сохранение в {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples_store, f, ensure_ascii=False, indent=2)
    print("Готово!")

if __name__ == "__main__":
    main()