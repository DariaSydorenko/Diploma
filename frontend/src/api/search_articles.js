const API_URL = 'http://localhost:8000/search_articles/';

export async function searchArticles(query, filters = {}) {
  const params = new URLSearchParams();
  params.append('query', query);

  if (filters.year) params.append('year', filters.year);
  if (filters.minCitations) params.append('min_citations', filters.minCitations);

  const response = await fetch(`${API_URL}?${params.toString()}`);

  if (!response.ok) {
    throw new Error('Помилка при пошуку');
  }
  return await response.json();
}
