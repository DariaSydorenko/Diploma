import React, { useState } from 'react';
import { searchArticles } from '../../api/search_articles';
import styles from "./SearchForm.module.css";

function SearchForm({ onResults, setLoading }) {
  const [query, setQuery] = useState('');
  const [year, setYear] = useState('');
  const [minCitations, setMinCitations] = useState('');
  const [localLoading, setLocalLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    onResults([], '');
    setLocalLoading(true);
    setLoading(true);

    try {
      const results = await searchArticles(query, { year, minCitations});
      onResults(results);
    } catch (err) {
      console.error('Помилка запиту:', err);
    }

    setLocalLoading(false);
    setLoading(false);
  };

  return (
    <form onSubmit={handleSubmit} className={styles.searchForm}>
      <input
        type="text"
        placeholder="Введіть науковий запит"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className={styles.input}
      />
      <input
        type="number"
        placeholder="Рік"
        value={year}
        onChange={(e) => setYear(e.target.value)}
        className={styles.input}
      />
      <input
        type="number"
        placeholder="Мін. цитувань"
        value={minCitations}
        onChange={(e) => setMinCitations(e.target.value)}
        className={styles.input}
      />

      <button type="submit" disabled={localLoading} className={styles.button}>
        {localLoading ? 'Пошук...' : 'Знайти'}
      </button>
    </form>
  );
}

export default SearchForm;
