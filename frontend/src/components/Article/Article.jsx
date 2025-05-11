import React from 'react';
import styles from './Article.module.css';

function Article({ article }) {
  return (
    <div className={styles.article}>
      <h3>
        <a href={article.doi} target="_blank" rel="noopener noreferrer">
          {article.display_name || 'Без назви'}
        </a>
      </h3>
      <p><strong>Рік:</strong> {article.publication_year}</p>
      <p><strong>Автори:</strong> {article.authors?.map(a => a.name).join(', ') || 'Невідомо'}</p>
      <p><strong>Цитованість:</strong> {article.cited_by_count}</p>
      <p><strong>DOI:</strong> {article.doi || '—'}</p>
      {/* <p><strong>Журнал:</strong> {article.journal?.display_name || '—'}</p> */}
    </div>
  );
}

export default Article;
