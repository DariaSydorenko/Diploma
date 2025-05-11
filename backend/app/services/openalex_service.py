import aiohttp
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from app.utils.text_processing import is_valid_work, full_text_from_work
import requests

async def fetch_openalex_data(session, base_url, params, cursor):
    """Оптимізоване отримання даних з OpenAlex API у асинхронному режимі"""
    try:
        # Додаємо User-Agent заголовок для запобігання 403 помилки
        headers = {
            "User-Agent": "Diploma/1.0 (mailto:dashasidorenko123@gmail.com)"
        }
        
        # Використовуємо cursor для пагінації
        request_params = {**params}
        if cursor and cursor != "*":
            request_params["cursor"] = cursor
        
        # Додаємо таймаут для запобігання зависанню запитів
        async with session.get(
            base_url, 
            params=request_params, 
            headers=headers, 
            timeout=15
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Помилка запиту до OpenAlex: {response.status}")
                response_text = await response.text()
                print(f"Текст відповіді: {response_text[:200]}")
                return {}
    except Exception as e:
        print(f"Помилка запиту до OpenAlex: {e}")
        return {}


from difflib import SequenceMatcher

def relevance_score(query, title, abstract):
    """Обчислює релевантність — більше значення = краща відповідність"""
    title_score = SequenceMatcher(None, query.lower(), title.lower()).ratio()
    abstract_score = SequenceMatcher(None, query.lower(), abstract.lower()).ratio() if abstract else 0
    return title_score * 2 + abstract_score  # заголовок має більшу вагу



async def parallel_fetch_openalex(query, base_url, params, max_results=100):
    """Оптимізоване паралельне отримання даних з OpenAlex"""
    next_cursor = "*"
    collected = 0
    all_results = []
    
    # Обмежуємо кількість паралельних запитів
    max_parallel_requests = 2  # Зменшуємо кількість паралельних запитів
    
    async with aiohttp.ClientSession() as session:
        # Перший запит для отримання початкових результатів
        first_response = await fetch_openalex_data(session, base_url, params, next_cursor)
        
        if not first_response or "results" not in first_response:
            return []
        
        results = first_response.get("results", [])
        
        scored_results = []
        for work in results:
            if is_valid_work(work):
                title = work.get("title", "")
                abstract = work.get("abstract_inverted_index", "")
                abstract_text = full_text_from_work(work)
                score = relevance_score(query, title, abstract_text)
                scored_results.append((score, (abstract_text, work)))

        # Сортуємо по релевантності
        scored_results.sort(reverse=True, key=lambda x: x[0])
        valid_results = [item for _, item in scored_results]

        all_results.extend(valid_results)
        collected += len(valid_results)
        
        # Отримання наступного курсора
        next_cursor = first_response.get("meta", {}).get("next_cursor")
        
        # Якщо нам потрібно більше результатів і є наступний курсор
        page_count = 1
        while next_cursor and collected < max_results and page_count < 5:  # Обмежуємо кількість сторінок
            page_count += 1
            
            try:
                # Отримуємо наступну сторінку результатів
                next_response = await fetch_openalex_data(session, base_url, params, next_cursor)
                
                if next_response and "results" in next_response:
                    results = next_response.get("results", [])
                    valid_results = [(full_text_from_work(work), work) for work in results if is_valid_work(work)]
                    all_results.extend(valid_results)
                    collected += len(valid_results)
                    
                    # Оновлюємо курсор для наступної сторінки
                    next_cursor = next_response.get("meta", {}).get("next_cursor")
                    if not next_cursor:
                        break
                else:
                    break
                    
            except Exception as e:
                print(f"Помилка при отриманні наступної сторінки: {e}")
                break
    
    # Обмежуємо результати до max_results
    return all_results[:max_results]
