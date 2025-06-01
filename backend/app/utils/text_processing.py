def is_valid_work(work):
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