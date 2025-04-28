import pandas as pd
import random
from category_attributes import category_attributes

# Генерация уникальных предметов
unique_items = set()
unique_dicts = []

for _ in range(5500):
    category = random.choice(list(category_attributes.keys()))
    attributes = category_attributes[category]["attributes"]
    columns = category_attributes[category]["columns"]
    selected_attributes = []
    for attr in attributes:
        if isinstance(attr, list):
            selected_attributes.append(random.choice(attr))
        else:
            selected_attributes.append(attr)
    num_attributes = random.randint(1, len(attributes))
    selected_attributes = random.sample(selected_attributes, num_attributes)
    selected_attributes_shuffled = selected_attributes
    random.shuffle(selected_attributes_shuffled)
    full_name = f"{category} {', '.join(selected_attributes_shuffled)}"
    # Создаем словарь для хранения характеристик
    item_dict = {"Полное название": full_name, "Категория": category}
    for i, attr in enumerate(attributes):
        attr_name = columns[i]
        if isinstance(attr, list):
            item_dict[attr_name] = next((a for a in attr if a in selected_attributes), "")
        else:
            item_dict[attr_name] = attr if attr in selected_attributes else ""
    # Проверка на уникальность
    item_tuple = tuple(item_dict.items())
    if item_tuple not in unique_items:
        unique_items.add(item_tuple)
        # Добавляем индекс 1 или 0 в зависимости от заполненности характеристик
        item_dict["Индекс заполненности"] = 1 if all(item_dict.get(col, "") for col in columns) else 0
        unique_dicts.append(item_dict)

# Создание DataFrame
df_final = pd.DataFrame(unique_dicts).fillna("")

# Сортировка данных
df_final_sorted = df_final.sort_values(by="Категория").reset_index(drop=True)

# Вставка пустых строк между разными категориями и добавление названий столбцов
df_with_gaps = []
previous_category = None

for _, row in df_final_sorted.iterrows():
    current_category = row["Категория"]
    if previous_category is None or previous_category != current_category:
        if previous_category is not None:
            # Добавляем пустую строку
            df_with_gaps.append([""] * len(df_final_sorted.columns))
        # Добавляем строку с названиями столбцов
        column_names_row = ["Полное название", "Категория"] + category_attributes[current_category]["columns"] + ["Индекс заполненности"]
        df_with_gaps.append(column_names_row)
    df_with_gaps.append([row.get(col, "") for col in ["Полное название", "Категория"] + category_attributes[current_category]["columns"] + ["Индекс заполненности"]])
    previous_category = current_category

# Создание нового DataFrame с разделителями
df_with_gaps = pd.DataFrame(df_with_gaps)

# Сохранение итогового файла
final_sorted_file_path = "dataset_generated.xlsx"
df_with_gaps.to_excel(final_sorted_file_path, index=False)

print(final_sorted_file_path)