def extract_abstract_text(inverted_index: dict) -> str:
    """Екстрагує повний текст абстракту з інвертованого індексу."""
    # if not abstract_index:
    #     return ""
    # try:
    #     words = [''] * (max(max(pos_list) for pos_list in abstract_index.values()) + 1)
    #     for word, positions in abstract_index.items():
    #         for pos in positions:
    #             words[pos] = word
    #     return ' '.join(words)
    # except Exception as e:
    #     print(f"Помилка при екстракції тексту з абстракту: {e}")
    #     return ""
    if not inverted_index:
        return ""
    sorted_words = sorted(inverted_index.items(), key=lambda x: min(x[1]))
    print(sorted_words)
    return " ".join(word for word, _ in sorted_words)
    

def is_valid_work(work):
    """Перевірка валідності роботи"""
    openalex_id = work.get("id")
    if not openalex_id:
        return False
        
    title = work.get("display_name", "")
    abstract = ""
    if isinstance(work.get("abstract_inverted_index"), dict):
        abstract = " ".join(work["abstract_inverted_index"].keys())
    full_text = f"{title} {abstract}".strip()
    
    return bool(full_text)


def full_text_from_work(work):
    """Отримання повного тексту для ембедінгу з правильною обробкою абстракту"""
    title = work.get("display_name", "")
    abstract = ""

    if isinstance(work.get("abstract_inverted_index"), dict):
        try:
            inverted_index = work["abstract_inverted_index"]
            positions = {}

            for word, positions_list in inverted_index.items():
                for pos in positions_list:
                    positions[pos] = word

            max_pos = max(positions.keys()) if positions else -1
            abstract_words = [positions.get(i, "") for i in range(max_pos + 1) if i in positions]
            abstract = " ".join(abstract_words)
        except Exception as e:
            print(f"Помилка обробки abstract_inverted_index: {e}")
            abstract = " ".join(work["abstract_inverted_index"].keys())
    
    return f"{title} {abstract}".strip()

# def full_text_from_article(work):
#     """Отримання повного тексту для ембедінгу з правильною обробкою абстракту"""
#     title = getattr(work, "display_name", "")
#     abstract = ""

#     if isinstance(getattr(work, "abstract_inverted_index", None), dict):
#         try:
#             inverted_index = getattr(work, "abstract_inverted_index", None)
#             positions = {}

#             for word, positions_list in inverted_index.items():
#                 for pos in positions_list:
#                     positions[pos] = word

#             max_pos = max(positions.keys()) if positions else -1
#             abstract_words = [positions.get(i, "") for i in range(max_pos + 1) if i in positions]
#             abstract = " ".join(abstract_words)
#         except Exception as e:
#             print(f"Помилка обробки abstract_inverted_index: {e}")
#             abstract = " ".join(abstract_words.keys())
    
#     return f"{title} {abstract}".strip()