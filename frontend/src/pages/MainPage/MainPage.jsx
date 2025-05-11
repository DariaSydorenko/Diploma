import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { signOut } from 'firebase/auth';
import { auth } from '../../firebaseConfig';
import SearchForm from '../../components/SearchForm/SearchForm';
import styles from './MainPage.module.css';
import Article from '../../components/Article/Article';

function MainPage() {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogout = async () => {
    await signOut(auth);
    navigate('/');
  };

  const handleResults = (results, source) => {
    setArticles(results);
    setLoading(false);
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1>Пошук наукових статей</h1>
        <button className={styles.logoutBtn} onClick={handleLogout}>
          🔓 Вийти
        </button>
      </div>

      <SearchForm
        onResults={handleResults}
        setLoading={(isLoading) => {
          setLoading(isLoading);
          if (isLoading) setArticles([]);
        }}
      />

      {loading ? (
        <p className={styles.message}>Виконується пошук...</p>
      ) : (
        <div className={styles.results}>
          {articles.length === 0 ? (
            <p className={styles.message}>Нічого не знайдено.</p>
          ) : (
            articles.map((article, i) => <Article key={i} article={article} />)
          )}
        </div>
      )}
    </div>
  );
}

export default MainPage;
