import React, { useState } from "react";
import { auth } from "../../firebaseConfig";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
} from "firebase/auth";
import { useNavigate } from "react-router-dom";
import styles from "./LoginPage.module.css";

const LoginPage = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isRegistering, setIsRegistering] = useState(false);
  const [message, setMessage] = useState("");
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      let userCredential;
      if (isRegistering) {
        userCredential = await createUserWithEmailAndPassword(auth, email, password);
      } else {
        userCredential = await signInWithEmailAndPassword(auth, email, password);
      }

      const idToken = await userCredential.user.getIdToken();
      const response = await fetch("http://localhost:8000/auth/verify-token", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${idToken}`,
        },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();
      if (response.ok) {
        navigate("/search_articles");
        setMessage(`✅ Успішно! Привіт, ${data.name || email}`);
      } else {
        setMessage(`✅ Успішна авторизація, але сервер відповів: ${data.detail}`);
      }
    } catch (err) {
      if (err.message.includes("Failed to fetch")) {
        setMessage("✅ Успішна авторизація через Firebase, але бекенд недоступний.");
      } else {
        setMessage("❌ Помилка: " + err.message);
      }
    }
  };

  return (
    <div className={styles.container}>
      <h2>{isRegistering ? "Реєстрація" : "Вхід"}</h2>
      <form onSubmit={handleSubmit} className={styles.form}>
        <input
          type="email"
          placeholder="Email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password"
          placeholder="Пароль"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button type="submit">
          {isRegistering ? "Зареєструватися" : "Увійти"}
        </button>
      </form>
      <p>
        {isRegistering ? "Вже є акаунт?" : "Немає акаунту?"}{" "}
        <button
          className={styles.toggleBtn}
          onClick={() => {
            setIsRegistering(!isRegistering);
            setMessage("");
          }}
        >
          {isRegistering ? "Увійти" : "Зареєструватися"}
        </button>
      </p>
      {message && <p className={styles.message}>{message}</p>}
    </div>
  );
};

export default LoginPage;
