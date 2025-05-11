import requests

def fetch_openalex_articles(query: str, filter_str: str = "", per_page: int = 25):
    url = f"https://api.openalex.org/works?search={query}&per-page={per_page}"

    if filter_str:
        url += f"&filter={filter_str}"

    url += "&select=id,doi,display_name,publication_year,updated_date,"
    "abstract_inverted_index,cited_by_count,concepts,keywords,"
    "open_access,has_fulltext,is_retracted,"
    "authorships,language,referenced_works,related_works"

    return requests.get(url)