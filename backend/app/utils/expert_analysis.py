from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util
import torch
import datetime
import numpy as np
from collections import Counter
import torch.nn.functional as F

model = SentenceTransformer('all-MiniLM-L6-v2')

MIN_ARTICLES_THRESHOLD = 5
SIMILARITY_YEAR_SPAN = 2
CURRENT_YEAR = datetime.datetime.now().year
MIN_RELEVANCE_THRESHOLD = 0.6
QUERY_LENGTH_THRESHOLD = 5

class Article:
    """Клас для представлення статті з усіма можливими полями від OpenAlex API."""
    def __init__(self, data: Dict[str, Any]):
        self.openalex_id = data.get("openalex_id")
        self.doi = data.get("doi")
        self.display_name = data.get("display_name")
        self.publication_year = data.get("publication_year")
        self.updated_date = data.get("updated_date")
        self.abstract_inverted_index = data.get("abstract_inverted_index")
        self.cited_by_count = data.get("cited_by_count")
        self.concepts = data.get("concepts")
        self.keywords = data.get("keywords")
        self.open_access = data.get("open_access")
        self.has_fulltext = data.get("has_fulltext")
        self.is_retracted = data.get("is_retracted")
        self.authorships = data.get("authorships")
        self.language = data.get("language")
        self.referenced_works = data.get("referenced_works")
        self.related_works = data.get("related_works")
        self.embedding = data.get("embedding")
        # self.relevance_score = data.get("relevance_score", 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертує об'єкт статті назад у словник для збереження в БД"""
        return {
            "openalex_id": self.openalex_id,
            "doi": self.doi,
            "display_name": self.display_name,
            "publication_year": self.publication_year,
            "updated_date": self.updated_date,
            "abstract_inverted_index": self.abstract_inverted_index,
            "cited_by_count": self.cited_by_count,
            "concepts": self.concepts,
            "keywords": self.keywords,
            "open_access": self.open_access,
            "has_fulltext": self.has_fulltext,
            "is_retracted": self.is_retracted,
            "authorships": self.authorships,
            "language": self.language,
            "referenced_works": self.referenced_works,
            "related_works": self.related_works,
            "embedding": self.embedding,
            # "relevance_score": self.relevance_score
        }

def compute_query_similarity(query: str, articles: List[Any]) -> float:
    """
    Обчислює середню семантичну схожість запиту до заголовків статей.
    
    Args:
        query: пошуковий запит
        articles: список статей
        
    Returns:
        Середній показник схожості (0.0-1.0)
    """
    if not articles:
        return 0.0
    
    # Кількість слів у запиті
    query_words = query.lower().split()
    
    # Для коротких запитів перевіряємо входження слів запиту в заголовки
    if len(query_words) < QUERY_LENGTH_THRESHOLD:
        match_scores = []
        for article in articles:
            title = (getattr(article, "display_name", "") or "").lower()
            if not title:
                continue
                
            # Рахуємо, яка частина слів запиту міститься в заголовку
            matches = sum(1 for word in query_words if word in title)
            match_ratio = matches / len(query_words) if query_words else 0
            match_scores.append(match_ratio)
        
        # Повертаємо середній показник входження слів запиту
        return sum(match_scores) / len(match_scores) if match_scores else 0.0
    
    # Для довших запитів використовуємо семантичну схожість
    try:
        # Кодуємо запит
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # Кодуємо заголовки статей
        titles = [getattr(article, "display_name", "") for article in articles if getattr(article, "display_name", "")]
        if not titles:
            return 0.0
            
        title_embeddings = model.encode(titles, convert_to_tensor=True)
        
        # Обчислюємо косинусну схожість між запитом і заголовками
        similarities = util.pytorch_cos_sim(query_embedding, title_embeddings)
        
        # Повертаємо середню схожість
        return torch.mean(similarities).item()
    except Exception as e:
        print(f"Помилка при обчисленні семантичної схожості: {e}")
        return 0.0

def is_results_insufficient(articles: List[Any], query: str = None) -> bool:
    """
    Перевіряє, чи достатньо результатів для повноцінного аналізу.
    
    Args:
        articles: список статей
        query: пошуковий запит (опціонально)
        
    Returns:
        True, якщо результатів недостатньо або вони нерелевантні
    """
    # Перевірка 1: Мінімальна кількість статей
    if len(articles) < MIN_ARTICLES_THRESHOLD:
        print(f"Недостатньо статей: {len(articles)} < {MIN_ARTICLES_THRESHOLD}")
        return True
    
    # Якщо запит не надано, ми не можемо перевірити релевантність
    if not query:
        # Перевірка 2: Різноманіття років публікації (тільки якщо запит не надано)
        years = [a.publication_year for a in articles if getattr(a, 'publication_year', None)]
        if years and len(years) >= 2 and (max(years) - min(years) <= SIMILARITY_YEAR_SPAN):
            print(f"Недостатнє різноманіття років: {min(years)}-{max(years)}")
            return True
    else:
        # Перевірка 3: Релевантність статей до запиту
        avg_similarity = compute_query_similarity(query, articles)
        if avg_similarity < MIN_RELEVANCE_THRESHOLD:
            print(f"Низька релевантність статей до запиту: {avg_similarity:.2f} < {MIN_RELEVANCE_THRESHOLD}")
            return True
        
        # Якщо релевантність висока, не перевіряємо різноманіття років
        # (релевантність важливіша за різноманіття)
        print(f"Висока релевантність статей до запиту: {avg_similarity:.2f}")
        return False
    
    # Якщо всі перевірки пройдені успішно
    return False

def extract_abstract_text(abstract_index: dict) -> str:
    """Екстрагує повний текст абстракту з інвертованого індексу."""
    if not abstract_index:
        return ""
    try:
        words = [''] * (max(max(pos_list) for pos_list in abstract_index.values()) + 1)
        for word, positions in abstract_index.items():
            for pos in positions:
                words[pos] = word
        return ' '.join(words)
    except Exception as e:
        print(f"Помилка при екстракції тексту з абстракту: {e}")
        return ""

def get_author_influence_score(authorships: List[Dict[str, Any]]) -> float:
    """Обчислює вплив авторів на основі їх h-індексу, affiliated_institutions та кількості публікацій."""
    if not authorships:
        return 0.0
    
    score = 0.0
    for author in authorships:
        # Базовий бал за автора
        author_score = 0.1
        
        # Враховуємо h-індекс автора, якщо він є
        author_h_index = author.get("author", {}).get("h_index", 0)
        if author_h_index:
            author_score += min(author_h_index / 50, 1.0) * 0.3
        
        # Враховуємо інституції
        institutions = author.get("institutions", [])
        if institutions:
            institution_count = len(institutions)
            author_score += min(institution_count * 0.05, 0.2)
            
            # Перевіряємо престижність інституцій (можна розширити)
            for inst in institutions:
                if inst.get("type") == "education" and inst.get("display_name"):
                    author_score += 0.05
        
        # Враховуємо роль автора (якщо головний автор)
        if author.get("author_position") in ["first", "corresponding"]:
            author_score *= 1.5
            
        score += author_score
    
    # Нормалізуємо фінальний бал
    return min(score, 1.0)

def get_concept_relevance_score(concepts: List[Dict[str, Any]], query: str) -> float:
    """Оцінює релевантність концептів статті до запиту."""
    if not concepts:
        return 0.0
    
    query_terms = set(query.lower().split())
    score = 0.0
    
    for concept in concepts:
        concept_name = concept.get("display_name", "").lower()
        
        # Перевірка на пряме співпадіння з запитом
        for term in query_terms:
            if term in concept_name:
                score += 0.15 * concept.get("score", 0)
        
        # Врахування загального рівня концепту
        level = concept.get("level", 0)
        if level == 0:  # Найвищий рівень концептів
            score += 0.1 * concept.get("score", 0)
        elif level == 1:
            score += 0.05 * concept.get("score", 0)
        else:
            score += 0.025 * concept.get("score", 0)
    
    return min(score, 1.0)

def get_citation_quality_score(cited_by_count: int, publication_year: int) -> float:
    """Оцінює якість цитування з урахуванням віку публікації."""
    if not cited_by_count or not publication_year:
        return 0.0
    
    years_since_publication = max(1, CURRENT_YEAR - publication_year)
    
    # Обчислюємо середню кількість цитувань за рік
    citations_per_year = cited_by_count / years_since_publication
    
    # Логарифмічна шкала для зменшення впливу статей з дуже високим цитуванням
    return min(0.3 * np.log1p(citations_per_year) / np.log1p(10), 0.3)

def get_keyword_match_score(keywords: List[Any], query: str) -> float:
    """Оцінює збіг ключових слів статті із запитом."""
    if not keywords:
        return 0.0
    
    query_terms = set(query.lower().split())
    matches = 0
    
    for keyword in keywords:
        # Обробляємо випадок, коли keyword може бути словником
        if isinstance(keyword, dict):
            # Шукаємо потрібні поля у словнику (display_name, name або value)
            keyword_text = keyword.get('display_name', keyword.get('name', keyword.get('value', '')))
            if isinstance(keyword_text, str):
                keyword_lower = keyword_text.lower()
            else:
                continue
        elif isinstance(keyword, str):
            keyword_lower = keyword.lower()
        else:
            continue
            
        for term in query_terms:
            if term in keyword_lower:
                matches += 1
    
    return min(matches * 0.1, 0.3)

def get_language_relevance(language: str, query: str) -> float:
    """Оцінює релевантність мови статті відносно запиту."""
    # Визначаємо мову запиту (спрощений підхід)
    query_lang = "en"  # За замовчуванням англійська
    
    # Спрощена логіка визначення мови запиту
    cyrillic_chars = set('абвгдеєжзиіїйклмнопрстуфхцчшщьюя')
    if any(c in cyrillic_chars for c in query.lower()):
        query_lang = "uk"
    
    # Якщо мова статті співпадає з мовою запиту або стаття англійською
    if language and ((language == query_lang) or (language == "en" and query_lang != "en")):
        return 0.1
    
    return 0.0

def compute_semantic_similarity(article: Article, query_embedding: torch.Tensor) -> float:
    """Обчислює семантичну схожість між статтею та запитом."""
    try:
        # Перевіряємо наявність ембедінгу в статті
        if hasattr(article, 'embedding') and article.embedding is not None:
            # Конвертація до тензору і нормалізація
            if isinstance(article.embedding, torch.Tensor):
                article_embedding = article.embedding
            elif isinstance(article.embedding, list) or isinstance(article.embedding, np.ndarray):
                article_embedding = torch.tensor(article.embedding, dtype=torch.float32)
            else:
                raise ValueError(f"Непідтримуваний тип ембедінгу: {type(article.embedding)}")
            
            # Нормалізація векторів для коректного косинусного порівняння
            query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
            article_norm = F.normalize(article_embedding.unsqueeze(0), p=2, dim=1)

            # Обчислення косинусної подібності
            similarity = torch.cosine_similarity(query_norm, article_norm).item()
        else:
            # Якщо ембедінг відсутній, генеруємо його з тексту
            title = getattr(article, 'display_name', "") or ""
            abstract = extract_abstract_text(getattr(article, 'abstract_inverted_index', {}))
            full_text = title + ". " + abstract
            
            if not full_text.strip():
                return 0.0
                
            article_embedding = model.encode(full_text, convert_to_tensor=True)
            # Нормалізація векторів
            query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
            article_norm = F.normalize(article_embedding.unsqueeze(0), p=2, dim=1)
            similarity = torch.cosine_similarity(query_norm, article_norm).item()
        
        return max(0.0, min(similarity, 1.0))  # Обмежуємо значення в діапазоні [0,1]
    except Exception as e:
        print(f"Помилка при обчисленні семантичної схожості: {e}")
        return 0.0

def compute_expert_score(article: Article, similarity: float, query: str) -> float:
    """Розрахунок комплексної експертної оцінки релевантності статті."""
    score = 0.0
    weights = {
        'semantic': 0.30,       # Семантична подібність
        'citation': 0.15,       # Якість цитування
        'recency': 0.10,        # Свіжість публікації
        'concepts': 0.15,       # Релевантність концептів
        'keywords': 0.10,       # Релевантність ключових слів
        'authorship': 0.10,     # Вплив авторів
        'accessibility': 0.05,  # Доступність
        'language': 0.05,       # Мовна релевантність
    }

    # 1. Семантична подібність (найважливіший фактор)
    score += similarity * weights['semantic']

    # 2. Якість цитування з урахуванням віку публікації
    if getattr(article, 'cited_by_count', None) and getattr(article, 'publication_year', None):
        score += get_citation_quality_score(article.cited_by_count, article.publication_year) * weights['citation']

    # 3. Свіжість публікації (актуальність)
    if getattr(article, 'publication_year', None):
        recency_score = max(0.0, 1.0 - max(0, (CURRENT_YEAR - article.publication_year)) / 10)
        score += recency_score * weights['recency']

    # 4. Релевантність концептів
    if getattr(article, 'concepts', None):
        score += get_concept_relevance_score(article.concepts, query) * weights['concepts']

    # 5. Релевантність ключових слів 
    if getattr(article, 'keywords', None):
        score += get_keyword_match_score(article.keywords, query) * weights['keywords']

    # 6. Авторський вплив
    if getattr(article, 'authorships', None):
        score += get_author_influence_score(article.authorships) * weights['authorship']

    # 7. Доступність
    accessibility_score = 0.0
    if getattr(article, "open_access", None) and article.open_access.get("is_oa"):
        accessibility_score += 0.6
    if getattr(article, "has_fulltext", False):
        accessibility_score += 0.4
    score += min(accessibility_score, 1.0) * weights['accessibility']

    # 8. Мовна релевантність
    if getattr(article, "language", None):
        score += get_language_relevance(article.language, query) * weights['language']

    # 9. Штрафи
    if getattr(article, "is_retracted", False):
        score *= 0.2  # Суттєвий штраф за відкликані статті

    return score

def analyze_articles(
        articles: List[Any], 
        query: str, 
        top_k: int, 
        raw_data: bool = True,
        relevance_threshold: float = 0.3
        ) -> List[Dict[str, Any]]:
    """
    Аналізує список статей і повертає top_k найбільш релевантних до запиту.
    
    Args:
        articles: список статей (словники або об'єкти Article)
        query: пошуковий запит
        top_k: кількість статей для повернення
        raw_data: якщо True, повертає словники замість об'єктів Article
        
    Returns:
        List[Dict[str, Any]]: список словників з даними статей, якщо raw_data=True
        або список об'єктів Article, якщо raw_data=False
    """
    if not articles:
        return []

    # Перевіряємо тип даних і перетворюємо відповідно
    article_objects = []
    for article_data in articles:
        if isinstance(article_data, dict):
            article_objects.append(Article(article_data))
        elif hasattr(article_data, 'openalex_id'):  # Це вже об'єкт Article
            article_objects.append(article_data)
        else:
            print(f"Невідомий тип даних статті: {type(article_data)}")
            continue
    
    # Створюємо вектор запиту
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Обчислюємо оцінки для кожної статті
    scored_articles = []
    for article in article_objects:
        try:
            semantic_similarity = compute_semantic_similarity(article, query_embedding)
            expert_score = compute_expert_score(article, semantic_similarity, query)
            scored_articles.append((article, expert_score))
        except Exception as e:
            print(f"Помилка при оцінці статті: {e}")
    
    # Сортуємо за оцінкою в спадному порядку
    scored_articles.sort(key=lambda x: x[1], reverse=True)
    
    filtered_articles = [(a, s) for a, s in scored_articles if s >= relevance_threshold]
    top_filtered = filtered_articles[:top_k]

    # Вивід у консоль результатів
    print(f"\n=== Результати аналізу для запиту: \"{query}\" ===")
    if not top_filtered:
        print("Не знайдено жодної релевантної статті вище порогу.")
    else:
        for idx, (article, score) in enumerate(top_filtered, 1):
            print(f"{idx}. {article.display_name} — Score: {score:.4f}")

    if raw_data:
        return [article.to_dict() for article, _ in top_filtered]
    else:
        return [article for article, _ in top_filtered]




















# from typing import List, Tuple, Dict, Any, Optional
# import re
# from collections import Counter
# import math
# import datetime
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.corpus import stopwords
# import nltk

# # Завантаження необхідних ресурсів NLTK
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')
# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     nltk.download('wordnet')

# # Константи
# MIN_ARTICLES_THRESHOLD = 5
# SIMILARITY_YEAR_SPAN = 2
# CURRENT_YEAR = datetime.datetime.now().year
# MIN_RELEVANCE_THRESHOLD = 0.5
# QUERY_LENGTH_THRESHOLD = 5

# # Ініціалізація інструментів для обробки тексту
# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))
# ukr_stop_words = set(['і', 'а', 'але', 'в', 'у', 'на', 'з', 'до', 'по', 'за', 'від', 'про'])
# stop_words.update(ukr_stop_words)

# # Додаємо україномовні стоп-слова
# # Розширюємо список при необхідності

# class Article:
#     """Клас для представлення статті з усіма можливими полями від OpenAlex API."""
#     def __init__(self, data: Dict[str, Any]):
#         self.openalex_id = data.get("openalex_id")
#         self.doi = data.get("doi")
#         self.display_name = data.get("display_name")
#         self.publication_year = data.get("publication_year")
#         self.updated_date = data.get("updated_date")
#         self.abstract_inverted_index = data.get("abstract_inverted_index")
#         self.cited_by_count = data.get("cited_by_count")
#         self.concepts = data.get("concepts")
#         self.keywords = data.get("keywords")
#         self.open_access = data.get("open_access")
#         self.has_fulltext = data.get("has_fulltext")
#         self.is_retracted = data.get("is_retracted")
#         self.authorships = data.get("authorships")
#         self.language = data.get("language") 
#         self.referenced_works = data.get("referenced_works")
#         self.related_works = data.get("related_works")
#         self._cached_content = None  # Кешований текст для швидшої роботи
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Конвертує об'єкт статті назад у словник для збереження в БД"""
#         return {
#             "openalex_id": self.openalex_id,
#             "doi": self.doi,
#             "display_name": self.display_name,
#             "publication_year": self.publication_year,
#             "updated_date": self.updated_date,
#             "abstract_inverted_index": self.abstract_inverted_index,
#             "cited_by_count": self.cited_by_count,
#             "concepts": self.concepts,
#             "keywords": self.keywords,
#             "open_access": self.open_access,
#             "has_fulltext": self.has_fulltext,
#             "is_retracted": self.is_retracted,
#             "authorships": self.authorships,
#             "language": self.language,
#             "referenced_works": self.referenced_works,
#             "related_works": self.related_works,
#         }
    
#     def get_content_for_analysis(self) -> str:
#         """Отримує текст статті для аналізу, включаючи назву та абстракт"""
#         if self._cached_content is not None:
#             return self._cached_content
            
#         title = self.display_name or ""
#         abstract = extract_abstract_text(self.abstract_inverted_index)
        
#         content = title
#         if abstract:
#             content += " " + abstract
        
#         self._cached_content = content    
#         return content
        
#     def preprocess_content(self) -> List[str]:
#         """Повертає передоброблений список токенів для аналізу"""
#         content = self.get_content_for_analysis()
#         return preprocess_text(content)

# def preprocess_text(text: str) -> List[str]:
#     """Функція для передобробки тексту - очищення, токенізація, видалення стоп-слів тощо"""
#     if not text:
#         return []
        
#     # Приведення до нижнього регістру
#     text = text.lower()
    
#     # Видалення пунктуації та цифр
#     text = re.sub(r'[^\w\s]', ' ', text)
#     text = re.sub(r'\d+', ' ', text)
    
#     # Токенізація
#     tokens = nltk.word_tokenize(text)
    
#     # Видалення стоп-слів
#     tokens = [token for token in tokens if token not in stop_words]
    
#     # Лематизація
#     lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
#     return lemmatized_tokens

# def extract_abstract_text(abstract_index: dict) -> str:
#     """Екстрагує повний текст абстракту з інвертованого індексу."""
#     if not abstract_index:
#         return ""
#     try:
#         words = [''] * (max(max(pos_list) for pos_list in abstract_index.values()) + 1)
#         for word, positions in abstract_index.items():
#             for pos in positions:
#                 words[pos] = word
#         return ' '.join(words)
#     except Exception as e:
#         print(f"Помилка при екстракції тексту з абстракту: {e}")
#         return ""

# def compute_tf_idf_similarity(query: str, articles: List[Article]) -> float:
#     """
#     Обчислює схожість запиту до статей за TF-IDF
#     """
#     if not articles or not query:
#         return 0.0
        
#     try:
#         # Отримуємо текст кожної статті
#         texts = [article.get_content_for_analysis() for article in articles]
#         if not texts:
#             return 0.0
            
#         # Додаємо запит до списку текстів для спільної векторизації
#         all_texts = [query] + texts
        
#         # Створюємо TF-IDF векторизатор
#         vectorizer = TfidfVectorizer(stop_words='english')
#         tfidf_matrix = vectorizer.fit_transform(all_texts)
        
#         # Отримуємо вектор запиту (перший елемент)
#         query_vector = tfidf_matrix[0]
        
#         # Обчислюємо косинусну схожість між запитом і кожною статтею
#         similarities = []
#         for i in range(1, len(all_texts)):
#             article_vector = tfidf_matrix[i]
#             # Косинусна схожість
#             similarity = (query_vector * article_vector.T).toarray()[0][0]
#             # Нормалізуємо у випадку нульових векторів
#             norm_query = np.sqrt((query_vector * query_vector.T).toarray()[0][0])
#             norm_article = np.sqrt((article_vector * article_vector.T).toarray()[0][0])
#             denominator = norm_query * norm_article
#             if denominator > 0:
#                 similarity /= denominator
#             similarities.append(similarity)
        
#         # Повертаємо середню схожість
#         avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
#         return avg_similarity
#     except Exception as e:
#         print(f"Помилка при обчисленні TF-IDF схожості: {e}")
#         return 0.0

# def compute_jaccard_similarity(query_terms: List[str], article_terms: List[str]) -> float:
#     """
#     Обчислює коефіцієнт Жаккара між термінами запиту та термінами статті
#     """
#     if not query_terms or not article_terms:
#         return 0.0
        
#     query_set = set(query_terms)
#     article_set = set(article_terms)
    
#     intersection = len(query_set.intersection(article_set))
#     union = len(query_set.union(article_set))
    
#     return intersection / union if union > 0 else 0.0

# def compute_bm25_score(query_terms: List[str], article_terms: List[str], 
#                        article_lengths: Dict[int, int], avg_length: float, 
#                        idf_scores: Dict[str, float], k1=1.5, b=0.75) -> float:
#     """
#     Обчислює оцінку BM25 для статті відносно запиту
#     """
#     if not query_terms or not article_terms:
#         return 0.0
        
#     score = 0.0
#     article_length = len(article_terms)
    
#     # Частоти термінів у статті
#     term_freqs = Counter(article_terms)
    
#     for term in query_terms:
#         if term not in idf_scores:
#             continue
            
#         idf = idf_scores[term]
#         tf = term_freqs.get(term, 0)
        
#         # Формула BM25
#         numerator = tf * (k1 + 1)
#         denominator = tf + k1 * (1 - b + b * article_length / avg_length)
#         score += idf * numerator / denominator if denominator > 0 else 0
        
#     return score

# def compute_query_similarity(query: str, articles: List[Article]) -> float:
#     """
#     Обчислює семантичну схожість запиту до заголовків статей.
#     Використовує спрощений алгоритм без ембедінгів.
    
#     Args:
#         query: пошуковий запит
#         articles: список статей
        
#     Returns:
#         Середній показник схожості (0.0-1.0)
#     """
#     if not articles:
#         return 0.0
    
#     # Кількість слів у запиті
#     query_words = query.lower().split()
#     preprocessed_query = preprocess_text(query)
    
#     # Для коротких запитів перевіряємо входження слів запиту в заголовки
#     if len(query_words) < QUERY_LENGTH_THRESHOLD:
#         match_scores = []
#         for article in articles:
#             title = (getattr(article, "display_name", "") or "").lower()
#             if not title:
#                 continue
                
#             # Рахуємо, яка частина слів запиту міститься в заголовку
#             matches = sum(1 for word in query_words if word in title)
#             match_ratio = matches / len(query_words) if query_words else 0
#             match_scores.append(match_ratio)
        
#         # Повертаємо середній показник входження слів запиту
#         return sum(match_scores) / len(match_scores) if match_scores else 0.0
    
#     # Для довших запитів використовуємо TF-IDF схожість
#     return compute_tf_idf_similarity(query, articles)

# def create_inverted_index(articles: List[Article]) -> Dict[str, List[int]]:
#     """
#     Створює інвертований індекс для колекції статей
#     """
#     inverted_index = {}
#     for idx, article in enumerate(articles):
#         terms = article.preprocess_content()
#         for term in set(terms):  # Використовуємо set для унікальних термінів
#             if term not in inverted_index:
#                 inverted_index[term] = []
#             inverted_index[term].append(idx)
#     return inverted_index

# def calculate_idf_scores(inverted_index: Dict[str, List[int]], total_docs: int) -> Dict[str, float]:
#     """
#     Обчислює IDF (Inverse Document Frequency) для кожного терміну
#     """
#     idf_scores = {}
#     for term, doc_ids in inverted_index.items():
#         # Кількість документів, що містять термін
#         df = len(doc_ids)
#         # IDF формула
#         idf_scores[term] = math.log((total_docs - df + 0.5) / (df + 0.5) + 1) if df > 0 else 0
#     return idf_scores

# def is_results_insufficient(articles: List[Any], query: str = None) -> bool:
#     """
#     Перевіряє, чи достатньо результатів для повноцінного аналізу.
    
#     Args:
#         articles: список статей
#         query: пошуковий запит (опціонально)
        
#     Returns:
#         True, якщо результатів недостатньо або вони нерелевантні
#     """
#     # Перевірка 1: Мінімальна кількість статей
#     if len(articles) < MIN_ARTICLES_THRESHOLD:
#         print(f"Недостатньо статей: {len(articles)} < {MIN_ARTICLES_THRESHOLD}")
#         return True
    
#     # Якщо запит не надано, ми не можемо перевірити релевантність
#     if not query:
#         # Перевірка 2: Різноманіття років публікації (тільки якщо запит не надано)
#         years = [a.publication_year for a in articles if getattr(a, 'publication_year', None)]
#         if years and len(years) >= 2 and (max(years) - min(years) <= SIMILARITY_YEAR_SPAN):
#             print(f"Недостатнє різноманіття років: {min(years)}-{max(years)}")
#             return True
#     else:
#         # Перевірка 3: Релевантність статей до запиту
#         avg_similarity = compute_query_similarity(query, articles)
#         if avg_similarity < MIN_RELEVANCE_THRESHOLD:
#             print(f"Низька релевантність статей до запиту: {avg_similarity:.2f} < {MIN_RELEVANCE_THRESHOLD}")
#             return True
        
#         # Якщо релевантність висока, не перевіряємо різноманіття років
#         # (релевантність важливіша за різноманіття)
#         print(f"Висока релевантність статей до запиту: {avg_similarity:.2f}")
#         return False
    
#     # Якщо всі перевірки пройдені успішно
#     return False

# def get_author_influence_score(authorships: List[Dict[str, Any]]) -> float:
#     """Обчислює вплив авторів на основі їх h-індексу, affiliated_institutions та кількості публікацій."""
#     if not authorships:
#         return 0.0
    
#     score = 0.0
#     for author in authorships:
#         # Базовий бал за автора
#         author_score = 0.1
        
#         # Враховуємо h-індекс автора, якщо він є
#         author_h_index = author.get("author", {}).get("h_index", 0)
#         if author_h_index:
#             # Збільшуємо вагу h-індексу
#             author_score += min(author_h_index / 40, 1.0) * 0.4
        
#         # Враховуємо інституції
#         institutions = author.get("institutions", [])
#         if institutions:
#             institution_count = len(institutions)
#             author_score += min(institution_count * 0.05, 0.2)
            
#             # Перевіряємо престижність інституцій (можна розширити)
#             for inst in institutions:
#                 if inst.get("type") == "education" and inst.get("display_name"):
#                     author_score += 0.05
        
#         # Враховуємо роль автора (якщо головний автор)
#         if author.get("author_position") in ["first", "corresponding"]:
#             author_score *= 1.5
            
#         score += author_score
    
#     # Нормалізуємо фінальний бал
#     return min(score, 1.0)

# def get_concept_relevance_score(concepts: List[Dict[str, Any]], query: str) -> float:
#     """Оцінює релевантність концептів статті до запиту - спрощена версія."""
#     if not concepts:
#         return 0.0
    
#     query_terms = set(preprocess_text(query))
#     score = 0.0
    
#     for concept in concepts:
#         concept_name = concept.get("display_name", "").lower()
#         concept_terms = set(preprocess_text(concept_name))
        
#         # Коефіцієнт Жаккара для концепта і запиту
#         if query_terms and concept_terms:
#             jaccard = compute_jaccard_similarity(list(query_terms), list(concept_terms))
#             score += jaccard * concept.get("score", 0) * 0.3
        
#         # Прямі входження термінів запиту в концепт
#         for term in query_terms:
#             if term in concept_terms:
#                 score += 0.2 * concept.get("score", 0)
        
#         # Врахування загального рівня концепту
#         level = concept.get("level", 0)
#         if level == 0:  # Найвищий рівень концептів
#             score += 0.15 * concept.get("score", 0)
#         elif level == 1:
#             score += 0.08 * concept.get("score", 0)
#         else:
#             score += 0.04 * concept.get("score", 0)
    
#     return min(score, 1.0)

# def get_citation_quality_score(cited_by_count: int, publication_year: int) -> float:
#     """Оцінює якість цитування з урахуванням віку публікації."""
#     if not cited_by_count or not publication_year:
#         return 0.0
    
#     years_since_publication = max(1, CURRENT_YEAR - publication_year)
    
#     # Обчислюємо середню кількість цитувань за рік
#     citations_per_year = cited_by_count / years_since_publication
    
#     if citations_per_year <= 1:
#         return 0.1 + 0.2 * citations_per_year  # Лінійне зростання до 1 цитування на рік
#     else:
#         # Логарифмічне зростання для більш як 1 цитування на рік
#         return 0.3 + 0.3 * np.log2(citations_per_year) / np.log2(20)
        
#     # Обмежуємо оцінку до 0.6 максимум
#     return min(score, 0.6)

# def get_keyword_match_score(keywords: List[Any], query: str) -> float:
#     """Оцінює збіг ключових слів статті із запитом."""
#     if not keywords:
#         return 0.0
    
#     query_terms = set(preprocess_text(query))
#     keyword_terms = []
    
#     # Обробляємо ключові слова
#     for keyword in keywords:
#         # Обробляємо випадок, коли keyword може бути словником
#         if isinstance(keyword, dict):
#             # Шукаємо потрібні поля у словнику (display_name, name або value)
#             keyword_text = keyword.get('display_name', keyword.get('name', keyword.get('value', '')))
#             if isinstance(keyword_text, str):
#                 keyword_lower = keyword_text.lower()
#                 keyword_terms.extend(preprocess_text(keyword_lower))
#         elif isinstance(keyword, str):
#             keyword_lower = keyword.lower()
#             keyword_terms.extend(preprocess_text(keyword_lower))
    
#     # Якщо нема термінів, повертаємо 0
#     if not keyword_terms:
#         return 0.0
    
#     # Обчислюємо коефіцієнт подібності Жаккара
#     jaccard = compute_jaccard_similarity(list(query_terms), keyword_terms)
    
#     # Рахуємо прямі входження
#     matches = 0
#     for term in query_terms:
#         if term in keyword_terms:
#             matches += 1
            
#     # Формуємо фінальну оцінку
#     if matches > 0 or jaccard > 0:
#         return min(0.1 + matches * 0.15 + jaccard * 0.3, 0.6)
    
#     return 0.0

# def get_language_relevance(language: str, query: str) -> float:
#     """Оцінює релевантність мови статті відносно запиту."""
#     # Визначаємо мову запиту (спрощений підхід)
#     query_lang = "en"  # За замовчуванням англійська
    
#     # Спрощена логіка визначення мови запиту
#     cyrillic_chars = set('абвгдеєжзиіїйклмнопрстуфхцчшщьюя')
#     if any(c in cyrillic_chars for c in query.lower()):
#         query_lang = "uk"
    
#     # Якщо мова статті співпадає з мовою запиту
#     if language and language == query_lang:
#         return 0.2
#     # Якщо стаття англійською, а запит іншою мовою (дозволяємо англійські статті)
#     elif language == "en" and query_lang != "en":
#         return 0.15
    
#     return 0.0

# def compute_semantic_similarity(article: Article, query: str) -> float:
#     """
#     Обчислює семантичну схожість між статтею та запитом.
#     Використовує спрощені методи без ембедінгів.
#     """
#     try:
#         article_content = article.get_content_for_analysis()
#         if not article_content:
#             return 0.0
            
#         # Підготовка термінів
#         query_terms = preprocess_text(query)
#         article_terms = preprocess_text(article_content)
        
#         if not query_terms or not article_terms:
#             return 0.0
            
#         # 1. Коефіцієнт подібності Жаккара
#         jaccard_sim = compute_jaccard_similarity(query_terms, article_terms)
        
#         # 2. Прямі збіги термінів (частка термінів запиту, що є у статті)
#         matches = sum(1 for term in query_terms if term in article_terms)
#         match_ratio = matches / len(query_terms) if query_terms else 0
        
#         # 3. TF-IDF схожість (якщо є достатньо термінів)
#         tfidf_sim = 0.0
#         if len(article_terms) > 10 and len(query_terms) > 2:
#             try:
#                 vectorizer = TfidfVectorizer()
#                 tfidf_matrix = vectorizer.fit_transform([' '.join(query_terms), ' '.join(article_terms)])
#                 tfidf_sim = (tfidf_matrix[0] * tfidf_matrix[1].T).toarray()[0][0]
#             except Exception:
#                 tfidf_sim = 0.0
        
#         # Комбінуємо різні метрики з ваговими коефіцієнтами
#         similarity = (0.4 * jaccard_sim) + (0.4 * match_ratio) + (0.2 * tfidf_sim)
        
#         # Коригуємо результат для кращого розподілу
#         boosted_similarity = max(0.0, min(1.0, similarity * 1.15))
        
#         return boosted_similarity
#     except Exception as e:
#         print(f"Помилка при обчисленні семантичної схожості: {e}")
#         return 0.0

# def compute_expert_score(article: Article, query: str) -> float:
#     """Розрахунок комплексної експертної оцінки релевантності статті."""
#     # Обчислюємо семантичну схожість для використання у вагах
#     semantic_similarity = compute_semantic_similarity(article, query)
    
#     # Переглянуті ваги для кращого розподілу оцінок
#     weights = {
#         'semantic': 0.40,       # Збільшено вагу семантичної подібності
#         'citation': 0.10,       # Якість цитування
#         'recency': 0.08,        # Свіжість публікації
#         'concepts': 0.15,       # Релевантність концептів
#         'keywords': 0.10,       # Релевантність ключових слів
#         'authorship': 0.07,     # Вплив авторів
#         'accessibility': 0.05,  # Доступність
#         'language': 0.05,       # Мовна релевантність
#     }

#     # Зберігаємо проміжні результати для діагностики
#     scores = {}

#     # 1. Семантична подібність (найважливіший фактор)
#     scores['semantic'] = semantic_similarity * weights['semantic']

#     # 2. Якість цитування з урахуванням віку публікації
#     if getattr(article, 'cited_by_count', None) and getattr(article, 'publication_year', None):
#         scores['citation'] = get_citation_quality_score(article.cited_by_count, article.publication_year) * weights['citation']
#     else:
#         scores['citation'] = 0.0

#     # 3. Свіжість публікації (актуальність)
#     if getattr(article, 'publication_year', None):
#         recency_score = max(0.0, 1.0 - max(0, (CURRENT_YEAR - article.publication_year)) / 10)
#         scores['recency'] = recency_score * weights['recency']
#     else:
#         scores['recency'] = 0.0

#     # 4. Релевантність концептів
#     if getattr(article, 'concepts', None):
#         scores['concepts'] = get_concept_relevance_score(article.concepts, query) * weights['concepts']
#     else:
#         scores['concepts'] = 0.0

#     # 5. Релевантність ключових слів 
#     if getattr(article, 'keywords', None):
#         scores['keywords'] = get_keyword_match_score(article.keywords, query) * weights['keywords']
#     else:
#         scores['keywords'] = 0.0

#     # 6. Авторський вплив
#     if getattr(article, 'authorships', None):
#         scores['authorship'] = get_author_influence_score(article.authorships) * weights['authorship']
#     else:
#         scores['authorship'] = 0.0

#     # 7. Доступність
#     accessibility_score = 0.0
#     if getattr(article, "open_access", None) and article.open_access.get("is_oa"):
#         accessibility_score += 0.6
#     if getattr(article, "has_fulltext", False):
#         accessibility_score += 0.4
#     scores['accessibility'] = min(accessibility_score, 1.0) * weights['accessibility']

#     # 8. Мовна релевантність
#     if getattr(article, "language", None):
#         scores['language'] = get_language_relevance(article.language, query) * weights['language']
#     else:
#         scores['language'] = 0.0

#     # Обчислюємо загальну оцінку
#     total_score = sum(scores.values())
    
#     # 9. Штрафи
#     if getattr(article, "is_retracted", False):
#         total_score *= 0.2  # Суттєвий штраф за відкликані статті

#     # Виводимо діагностичну інформацію
#     print(f"\nДіагностика для статті: {article.display_name}")
#     for component, score in scores.items():
#         print(f"  - {component}: {score:.4f}")
#     print(f"  = Загальна оцінка: {total_score:.4f}")

#     return total_score

# def analyze_articles(
#         articles: List[Any], 
#         query: str, 
#         top_k: int, 
#         raw_data: bool = True,
#         relevance_threshold: float = 0.3
#         ) -> List[Dict[str, Any]]:
#     """
#     Аналізує список статей і повертає top_k найбільш релевантних до запиту.
    
#     Args:
#         articles: список статей (словники або об'єкти Article)
#         query: пошуковий запит
#         top_k: кількість статей для повернення
#         raw_data: якщо True, повертає словники замість об'єктів Article
        
#     Returns:
#         List[Dict[str, Any]]: список словників з даними статей, якщо raw_data=True
#         або список об'єктів Article, якщо raw_data=False
#     """
#     if not articles:
#         return []

#     # Перевіряємо тип даних і перетворюємо відповідно
#     article_objects = []
#     for article_data in articles:
#         if isinstance(article_data, dict):
#             article_objects.append(Article(article_data))
#         elif hasattr(article_data, 'openalex_id'):  # Це вже об'єкт Article
#             article_objects.append(article_data)
#         else:
#             print(f"Невідомий тип даних статті: {type(article_data)}")
#             continue
    
#     # Створюємо вектор запиту
#     query_embedding = model.encode(query, convert_to_tensor=True)
    
#     # Обчислюємо оцінки для кожної статті
#     scored_articles = []
#     for article in article_objects:
#         try:
#             semantic_similarity = compute_semantic_similarity(article, query_embedding)
#             expert_score = compute_expert_score(article, semantic_similarity, query)
#             scored_articles.append((article, expert_score))
#         except Exception as e:
#             print(f"Помилка при оцінці статті: {e}")
    
#     # Сортуємо за оцінкою в спадному порядку
#     scored_articles.sort(key=lambda x: x[1], reverse=True)
    
#     filtered_articles = [(a, s) for a, s in scored_articles if s >= relevance_threshold]
#     top_filtered = filtered_articles[:top_k]

#     # Вивід у консоль результатів
#     print(f"\n=== Результати аналізу для запиту: \"{query}\" ===")
#     if not top_filtered:
#         print("Не знайдено жодної релевантної статті вище порогу.")
#     else:
#         for idx, (article, score) in enumerate(top_filtered, 1):
#             print(f"{idx}. {article.display_name} — Score: {score:.4f}")

#     if raw_data:
#         return [article.to_dict() for article, _ in top_filtered]
#     else:
#         return [article for article, _ in top_filtered]
















# from typing import List, Tuple, Dict, Any, Optional
# from sentence_transformers import SentenceTransformer, util
# import torch
# import datetime
# import numpy as np
# from collections import Counter
# import torch.nn.functional as F
# import functools

# # Використовуємо модель з кешуванням для покращення швидкості
# model = SentenceTransformer('all-mpnet-base-v2')

# MIN_ARTICLES_THRESHOLD = 5
# SIMILARITY_YEAR_SPAN = 2
# CURRENT_YEAR = datetime.datetime.now().year
# MIN_RELEVANCE_THRESHOLD = 0.6
# QUERY_LENGTH_THRESHOLD = 5

# # Додаємо кешування для ембедінгів запитів
# query_embedding_cache = {}

# class Article:
#     """Клас для представлення статті з усіма можливими полями від OpenAlex API."""
#     def __init__(self, data: Dict[str, Any]):
#         self.openalex_id = data.get("openalex_id")
#         self.doi = data.get("doi")
#         self.display_name = data.get("display_name")
#         self.publication_year = data.get("publication_year")
#         self.updated_date = data.get("updated_date")
#         self.abstract_inverted_index = data.get("abstract_inverted_index")
#         self.cited_by_count = data.get("cited_by_count")
#         self.concepts = data.get("concepts")
#         self.keywords = data.get("keywords")
#         self.open_access = data.get("open_access")
#         self.has_fulltext = data.get("has_fulltext")
#         self.is_retracted = data.get("is_retracted")
#         self.authorships = data.get("authorships")
#         self.language = data.get("language")
#         self.referenced_works = data.get("referenced_works")
#         self.related_works = data.get("related_works")
#         self.embedding = data.get("embedding")
#         self._cached_embedding = None  # Кешований ембедінг для швидшої роботи
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Конвертує об'єкт статті назад у словник для збереження в БД"""
#         return {
#             "openalex_id": self.openalex_id,
#             "doi": self.doi,
#             "display_name": self.display_name,
#             "publication_year": self.publication_year,
#             "updated_date": self.updated_date,
#             "abstract_inverted_index": self.abstract_inverted_index,
#             "cited_by_count": self.cited_by_count,
#             "concepts": self.concepts,
#             "keywords": self.keywords,
#             "open_access": self.open_access,
#             "has_fulltext": self.has_fulltext,
#             "is_retracted": self.is_retracted,
#             "authorships": self.authorships,
#             "language": self.language,
#             "referenced_works": self.referenced_works,
#             "related_works": self.related_works,
#             "embedding": self.embedding,
#         }
    
#     def get_content_for_embedding(self) -> str:
#         """Отримує текст статті для ембедінгу, включаючи назву та абстракт"""
#         title = self.display_name or ""
#         abstract = extract_abstract_text(self.abstract_inverted_index)
        
#         content = title
#         if abstract:
#             content += " " + abstract
            
#         return content
    
#     def get_embedding(self, force_recompute=False) -> Optional[torch.Tensor]:
#         """Отримує ембедінг статті, генеруючи його при необхідності"""
#         if self._cached_embedding is not None and not force_recompute:
#             return self._cached_embedding
            
#         # Використовуємо існуючий ембедінг якщо є
#         if hasattr(self, 'embedding') and self.embedding is not None:
#             if isinstance(self.embedding, torch.Tensor):
#                 self._cached_embedding = self.embedding
#             elif isinstance(self.embedding, list) or isinstance(self.embedding, np.ndarray):
#                 self._cached_embedding = torch.tensor(self.embedding, dtype=torch.float32)
#             return self._cached_embedding
        
#         # Інакше генеруємо новий ембедінг
#         content = self.get_content_for_embedding()
#         if not content.strip():
#             return None
            
#         try:
#             self._cached_embedding = model.encode(content, convert_to_tensor=True)
#             return self._cached_embedding
#         except Exception as e:
#             print(f"Помилка при генерації ембедінгу: {e}")
#             return None

# # Функція з кешуванням для покращення швидкодії
# @functools.lru_cache(maxsize=128)
# def get_query_embedding(query: str) -> torch.Tensor:
#     """Отримує ембедінг запиту з кешуванням для повторних запитів"""
#     return model.encode(query, convert_to_tensor=True)

# def compute_query_similarity(query: str, articles: List[Any]) -> float:
#     """
#     Обчислює середню семантичну схожість запиту до заголовків статей.
    
#     Args:
#         query: пошуковий запит
#         articles: список статей
        
#     Returns:
#         Середній показник схожості (0.0-1.0)
#     """
#     if not articles:
#         return 0.0
    
#     # Кількість слів у запиті
#     query_words = query.lower().split()
    
#     # Для коротких запитів перевіряємо входження слів запиту в заголовки
#     if len(query_words) < QUERY_LENGTH_THRESHOLD:
#         match_scores = []
#         for article in articles:
#             title = (getattr(article, "display_name", "") or "").lower()
#             if not title:
#                 continue
                
#             # Рахуємо, яка частина слів запиту міститься в заголовку
#             matches = sum(1 for word in query_words if word in title)
#             match_ratio = matches / len(query_words) if query_words else 0
#             match_scores.append(match_ratio)
        
#         # Повертаємо середній показник входження слів запиту
#         return sum(match_scores) / len(match_scores) if match_scores else 0.0
    
#     # Для довших запитів використовуємо семантичну схожість
#     try:
#         # Кодуємо запит з використанням кешування
#         query_embedding = get_query_embedding(query)
        
#         # Кодуємо заголовки статей
#         titles = [getattr(article, "display_name", "") for article in articles if getattr(article, "display_name", "")]
#         if not titles:
#             return 0.0
            
#         title_embeddings = model.encode(titles, convert_to_tensor=True)
        
#         # Обчислюємо косинусну схожість між запитом і заголовками
#         similarities = util.cos_sim(query_embedding, title_embeddings)
        
#         # Повертаємо середню схожість
#         return torch.mean(similarities).item()
#     except Exception as e:
#         print(f"Помилка при обчисленні семантичної схожості: {e}")
#         return 0.0

# def is_results_insufficient(articles: List[Any], query: str = None) -> bool:
#     """
#     Перевіряє, чи достатньо результатів для повноцінного аналізу.
    
#     Args:
#         articles: список статей
#         query: пошуковий запит (опціонально)
        
#     Returns:
#         True, якщо результатів недостатньо або вони нерелевантні
#     """
#     # Перевірка 1: Мінімальна кількість статей
#     if len(articles) < MIN_ARTICLES_THRESHOLD:
#         print(f"Недостатньо статей: {len(articles)} < {MIN_ARTICLES_THRESHOLD}")
#         return True
    
#     # Якщо запит не надано, ми не можемо перевірити релевантність
#     if not query:
#         # Перевірка 2: Різноманіття років публікації (тільки якщо запит не надано)
#         years = [a.publication_year for a in articles if getattr(a, 'publication_year', None)]
#         if years and len(years) >= 2 and (max(years) - min(years) <= SIMILARITY_YEAR_SPAN):
#             print(f"Недостатнє різноманіття років: {min(years)}-{max(years)}")
#             return True
#     else:
#         # Перевірка 3: Релевантність статей до запиту
#         avg_similarity = compute_query_similarity(query, articles)
#         if avg_similarity < MIN_RELEVANCE_THRESHOLD:
#             print(f"Низька релевантність статей до запиту: {avg_similarity:.2f} < {MIN_RELEVANCE_THRESHOLD}")
#             return True
        
#         # Якщо релевантність висока, не перевіряємо різноманіття років
#         # (релевантність важливіша за різноманіття)
#         print(f"Висока релевантність статей до запиту: {avg_similarity:.2f}")
#         return False
    
#     # Якщо всі перевірки пройдені успішно
#     return False

# def extract_abstract_text(abstract_index: dict) -> str:
#     """Екстрагує повний текст абстракту з інвертованого індексу."""
#     if not abstract_index:
#         return ""
#     try:
#         words = [''] * (max(max(pos_list) for pos_list in abstract_index.values()) + 1)
#         for word, positions in abstract_index.items():
#             for pos in positions:
#                 words[pos] = word
#         return ' '.join(words)
#     except Exception as e:
#         print(f"Помилка при екстракції тексту з абстракту: {e}")
#         return ""

# # Додаємо кешування для часто використовуваних функцій
# @functools.lru_cache(maxsize=128)
# def get_author_influence_score(author_tuple: tuple) -> float:
#     """Обчислює вплив авторів на основі їх h-індексу, affiliated_institutions та кількості публікацій."""
#     # Перетворюємо tuple назад на список словників
#     authorships = list(author_tuple)
    
#     if not authorships:
#         return 0.0
    
#     score = 0.0
#     for author in authorships:
#         # Базовий бал за автора
#         author_score = 0.1
        
#         # Враховуємо h-індекс автора, якщо він є
#         author_h_index = author.get("author", {}).get("h_index", 0)
#         if author_h_index:
#             # Збільшуємо вагу h-індексу
#             author_score += min(author_h_index / 40, 1.0) * 0.4
        
#         # Враховуємо інституції
#         institutions = author.get("institutions", [])
#         if institutions:
#             institution_count = len(institutions)
#             author_score += min(institution_count * 0.05, 0.2)
            
#             # Перевіряємо престижність інституцій (можна розширити)
#             for inst in institutions:
#                 if inst.get("type") == "education" and inst.get("display_name"):
#                     author_score += 0.05
        
#         # Враховуємо роль автора (якщо головний автор)
#         if author.get("author_position") in ["first", "corresponding"]:
#             author_score *= 1.5
            
#         score += author_score
    
#     # Нормалізуємо фінальний бал
#     return min(score, 1.0)

# def get_concept_relevance_score(concepts: List[Dict[str, Any]], query: str) -> float:
#     """Оцінює релевантність концептів статті до запиту."""
#     if not concepts:
#         return 0.0
    
#     query_terms = set(query.lower().split())
#     score = 0.0
    
#     # Створюємо ембедінг запиту з кешуванням
#     try:
#         query_embedding = get_query_embedding(query)
        
#         for concept in concepts:
#             concept_name = concept.get("display_name", "").lower()
            
#             # Перевірка на пряме співпадіння з запитом
#             for term in query_terms:
#                 if term in concept_name:
#                     # Збільшуємо ваги для прямих співпадінь
#                     score += 0.2 * concept.get("score", 0)
            
#             # Додаємо семантичне порівняння концептів
#             try:
#                 # Використовуємо кешовану функцію для отримання ембедінгу концепту
#                 concept_embedding = get_query_embedding(concept_name)
                
#                 # Виправлення проблеми з розмірністю тензорів
#                 # Переконуємося, що вектори мають правильну форму перед множенням
#                 if query_embedding.dim() == 1:
#                     query_embedding_reshaped = query_embedding.unsqueeze(0)
#                 else:
#                     query_embedding_reshaped = query_embedding
                    
#                 if concept_embedding.dim() == 1:
#                     concept_embedding_reshaped = concept_embedding.unsqueeze(0)
#                 else: 
#                     concept_embedding_reshaped = concept_embedding
                
#                 # Обчислюємо косинусну схожість
#                 similarity = util.cos_sim(query_embedding_reshaped, concept_embedding_reshaped)[0][0].item()
#                 score += 0.15 * similarity * concept.get("score", 0)
#             except Exception as e:
#                 print(f"Помилка при порівнянні концепту {concept_name}: {e}")
            
#             # Врахування загального рівня концепту
#             level = concept.get("level", 0)
#             if level == 0:  # Найвищий рівень концептів
#                 score += 0.15 * concept.get("score", 0)
#             elif level == 1:
#                 score += 0.08 * concept.get("score", 0)
#             else:
#                 score += 0.04 * concept.get("score", 0)
#     except Exception as e:
#         print(f"Помилка при аналізі концептів: {e}")
        
#         # Запасний варіант без семантичного порівняння
#         for concept in concepts:
#             concept_name = concept.get("display_name", "").lower()
            
#             # Перевірка на пряме співпадіння з запитом
#             for term in query_terms:
#                 if term in concept_name:
#                     score += 0.2 * concept.get("score", 0)
            
#             # Врахування загального рівня концепту
#             level = concept.get("level", 0)
#             if level == 0:
#                 score += 0.15 * concept.get("score", 0)
#             elif level == 1:
#                 score += 0.08 * concept.get("score", 0)
#             else:
#                 score += 0.04 * concept.get("score", 0)
    
#     return min(score, 1.0)

# @functools.lru_cache(maxsize=128)
# def get_citation_quality_score(cited_by_count: int, publication_year: int) -> float:
#     """Оцінює якість цитування з урахуванням віку публікації."""
#     if not cited_by_count or not publication_year:
#         return 0.0
    
#     years_since_publication = max(1, CURRENT_YEAR - publication_year)
    
#     # Обчислюємо середню кількість цитувань за рік
#     citations_per_year = cited_by_count / years_since_publication
    
#     # Покращена формула для кращого розподілу балів
#     if citations_per_year <= 1:
#         score = 0.1 + 0.2 * citations_per_year  # Лінійне зростання до 1 цитування на рік
#     else:
#         # Логарифмічне зростання для більш як 1 цитування на рік
#         score = 0.3 + 0.3 * np.log2(citations_per_year) / np.log2(20)
        
#     # Обмежуємо оцінку до 0.6 максимум
#     return min(score, 0.6)

# def get_keyword_match_score(keywords: List[Any], query: str) -> float:
#     """Оцінює збіг ключових слів статті із запитом."""
#     if not keywords:
#         return 0.0
    
#     query_terms = set(query.lower().split())
#     matches = 0
    
#     # Підраховуємо кількість збігів ключових слів з запитом
#     for keyword in keywords:
#         # Обробляємо випадок, коли keyword може бути словником
#         if isinstance(keyword, dict):
#             # Шукаємо потрібні поля у словнику (display_name, name або value)
#             keyword_text = keyword.get('display_name', keyword.get('name', keyword.get('value', '')))
#             if isinstance(keyword_text, str):
#                 keyword_lower = keyword_text.lower()
#             else:
#                 continue
#         elif isinstance(keyword, str):
#             keyword_lower = keyword.lower()
#         else:
#             continue
            
#         for term in query_terms:
#             if term in keyword_lower:
#                 matches += 1
                
#                 # Додаємо бонус за точний збіг
#                 if term == keyword_lower:
#                     matches += 0.5
    
#     # Покращуємо оцінку для кращого розподілу
#     if matches > 0:
#         return min(0.1 + matches * 0.15, 0.6)
#     return 0.0

# @functools.lru_cache(maxsize=128)
# def get_language_relevance(language: str, query: str) -> float:
#     """Оцінює релевантність мови статті відносно запиту."""
#     # Визначаємо мову запиту (спрощений підхід)
#     query_lang = "en"  # За замовчуванням англійська
    
#     # Спрощена логіка визначення мови запиту
#     cyrillic_chars = set('абвгдеєжзиіїйклмнопрстуфхцчшщьюя')
#     if any(c in cyrillic_chars for c in query.lower()):
#         query_lang = "uk"
    
#     # Якщо мова статті співпадає з мовою запиту
#     if language and language == query_lang:
#         return 0.2
#     # Якщо стаття англійською, а запит іншою мовою (дозволяємо англійські статті)
#     elif language == "en" and query_lang != "en":
#         return 0.15
    
#     return 0.0

# def compute_semantic_similarity(article: Article, query_embedding: torch.Tensor) -> float:
#     """Обчислює семантичну схожість між статтею та запитом."""
#     try:
#         # Отримуємо ембедінг статті
#         article_embedding = article.get_embedding()
        
#         if article_embedding is None:
#             return 0.0
        
#         # ВИПРАВЛЕННЯ: переконуємося, що вектори мають правильну форму перед множенням
#         if query_embedding.dim() == 1:
#             query_embedding_reshaped = query_embedding.unsqueeze(0)
#         else:
#             query_embedding_reshaped = query_embedding
            
#         if article_embedding.dim() == 1:
#             article_embedding_reshaped = article_embedding.unsqueeze(0)
#         else: 
#             article_embedding_reshaped = article_embedding
        
#         # Обчислення косинусної подібності з правильними розмірностями
#         similarity = util.cos_sim(query_embedding_reshaped, article_embedding_reshaped)[0][0].item()
            
#         # Збільшуємо вагу схожості для покращення розподілу оцінок
#         boosted_similarity = max(0.0, min(1.0, similarity * 1.15))
        
#         return boosted_similarity
#     except Exception as e:
#         print(f"Помилка при обчисленні семантичної схожості: {e}")
#         return 0.0

# def compute_expert_score(article: Article, similarity: float, query: str) -> float:
#     """Розрахунок комплексної експертної оцінки релевантності статті."""
#     # Переглянуті ваги для кращого розподілу оцінок
#     weights = {
#         'semantic': 0.40,       # Збільшено вагу семантичної подібності
#         'citation': 0.10,       # Якість цитування
#         'recency': 0.08,        # Свіжість публікації
#         'concepts': 0.15,       # Релевантність концептів
#         'keywords': 0.10,       # Релевантність ключових слів
#         'authorship': 0.07,     # Вплив авторів
#         'accessibility': 0.05,  # Доступність
#         'language': 0.05,       # Мовна релевантність
#     }

#     # Зберігаємо проміжні результати для діагностики
#     scores = {}

#     # 1. Семантична подібність (найважливіший фактор)
#     scores['semantic'] = similarity * weights['semantic']

#     # 2. Якість цитування з урахуванням віку публікації
#     if getattr(article, 'cited_by_count', None) and getattr(article, 'publication_year', None):
#         scores['citation'] = get_citation_quality_score(article.cited_by_count, article.publication_year) * weights['citation']
#     else:
#         scores['citation'] = 0.0

#     # 3. Свіжість публікації (актуальність)
#     if getattr(article, 'publication_year', None):
#         recency_score = max(0.0, 1.0 - max(0, (CURRENT_YEAR - article.publication_year)) / 10)
#         scores['recency'] = recency_score * weights['recency']
#     else:
#         scores['recency'] = 0.0

#     # 4. Релевантність концептів
#     if getattr(article, 'concepts', None):
#         scores['concepts'] = get_concept_relevance_score(article.concepts, query) * weights['concepts']
#     else:
#         scores['concepts'] = 0.0

#     # 5. Релевантність ключових слів 
#     if getattr(article, 'keywords', None):
#         scores['keywords'] = get_keyword_match_score(article.keywords, query) * weights['keywords']
#     else:
#         scores['keywords'] = 0.0

#     # 6. Авторський вплив
#     if getattr(article, 'authorships', None):
#         # Перетворення списку словників у хешовану структуру для кешування
#         authorships_tuple = tuple(tuple(sorted(author.items())) for author in article.authorships)
#         scores['authorship'] = 0.0  # Значення за замовчуванням
        
#         # Оптимізовано: обчислюємо лише якщо є авторство
#         if authorships_tuple:
#             try:
#                 scores['authorship'] = get_author_influence_score(authorships_tuple) * weights['authorship']
#             except:
#                 # Запасний варіант: спрощений розрахунок
#                 scores['authorship'] = min(0.4, len(article.authorships) * 0.07) * weights['authorship']
#     else:
#         scores['authorship'] = 0.0

#     # 7. Доступність
#     accessibility_score = 0.0
#     if getattr(article, "open_access", None) and article.open_access.get("is_oa"):
#         accessibility_score += 0.6
#     if getattr(article, "has_fulltext", False):
#         accessibility_score += 0.4
#     scores['accessibility'] = min(accessibility_score, 1.0) * weights['accessibility']

#     # 8. Мовна релевантність
#     if getattr(article, "language", None):
#         scores['language'] = get_language_relevance(article.language, query) * weights['language']
#     else:
#         scores['language'] = 0.0

#     # Обчислюємо загальну оцінку
#     total_score = sum(scores.values())
    
#     # 9. Штрафи
#     if getattr(article, "is_retracted", False):
#         total_score *= 0.2  # Суттєвий штраф за відкликані статті

#     # Виводимо діагностичну інформацію (за потреби можна відключити для підвищення швидкодії)
#     # print(f"\nДіагностика для статті: {article.display_name}")
#     # for component, score in scores.items():
#     #     print(f"  - {component}: {score:.4f}")
#     # print(f"  = Загальна оцінка: {total_score:.4f}")

#     return total_score

# def analyze_articles(
#         articles: List[Any], 
#         query: str, 
#         top_k: int, 
#         raw_data: bool = True,
#         relevance_threshold: float = 0.3
#         ) -> List[Dict[str, Any]]:
#     """
#     Аналізує список статей і повертає top_k найбільш релевантних до запиту.
    
#     Args:
#         articles: список статей (словники або об'єкти Article)
#         query: пошуковий запит
#         top_k: кількість статей для повернення
#         raw_data: якщо True, повертає словники замість об'єктів Article
        
#     Returns:
#         List[Dict[str, Any]]: список словників з даними статей, якщо raw_data=True
#         або список об'єктів Article, якщо raw_data=False
#     """
#     if not articles:
#         return []

#     # Оптимізація: перетворюємо список статей для паралельної обробки
#     article_objects = []
#     for article_data in articles:
#         if isinstance(article_data, dict):
#             article_objects.append(Article(article_data))
#         elif hasattr(article_data, 'openalex_id'):  # Це вже об'єкт Article
#             article_objects.append(article_data)
#         else:
#             print(f"Невідомий тип даних статті: {type(article_data)}")
#             continue
    
#     # Оптимізація: використовуємо кешування для запиту
#     query_embedding = get_query_embedding(query)
    
#     # Обчислюємо оцінки для кожної статті
#     scored_articles = []
#     for article in article_objects:
#         try:
#             semantic_similarity = compute_semantic_similarity(article, query_embedding)
#             expert_score = compute_expert_score(article, semantic_similarity, query)
#             scored_articles.append((article, expert_score))
#         except Exception as e:
#             print(f"Помилка при оцінці статті: {e}")
    
#     # Сортуємо за оцінкою в спадному порядку
#     scored_articles.sort(key=lambda x: x[1], reverse=True)
    
#     filtered_articles = [(a, s) for a, s in scored_articles if s >= relevance_threshold]
#     top_filtered = filtered_articles[:top_k]

#     # Вивід у консоль результатів
#     print(f"\n=== Результати аналізу для запиту: \"{query}\" ===")
#     if not top_filtered:
#         print("Не знайдено жодної релевантної статті вище порогу.")
#     else:
#         for idx, (article, score) in enumerate(top_filtered, 1):
#             print(f"{idx}. {article.display_name} — Score: {score:.4f}")

#     if raw_data:
#         return [article.to_dict() for article, _ in top_filtered]
#     else:
#         return [article for article, _ in top_filtered]