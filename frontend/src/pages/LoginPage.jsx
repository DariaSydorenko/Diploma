import React, { useState } from "react";
import { auth } from "../firebaseConfig";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
} from "firebase/auth";

const LoginPage = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isRegistering, setIsRegistering] = useState(false);
  const [message, setMessage] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // 🔐 Увійти або зареєструватися
      let userCredential;
      if (isRegistering) {
        userCredential = await createUserWithEmailAndPassword(auth, email, password);
      } else {
        userCredential = await signInWithEmailAndPassword(auth, email, password);
      }
    
      // 🔑 Отримати токен
      const idToken = await userCredential.user.getIdToken();
    
      // 📡 Надіслати токен на бекенд
      const response = await fetch("http://localhost:8000/auth/verify-token", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${idToken}`,
        },
        body: JSON.stringify({ email }),
      });
    
      const data = await response.json();

      console.log(response)
    
      if (response.ok) {
        setMessage(`✅ Успішно! Привіт, ${data.name || email}`);
      } else {
        // 👉 навіть якщо сервер відповів помилкою — показати хоча б успішний вхід
        setMessage(`✅ Успішна авторизація через Firebase, але сервер відповів з помилкою: ${data.detail}`);
      }
    } catch (err) {
      console.error(err);
    
      // 🎯 Якщо помилка — перевіримо, чи це саме fetch, чи щось інше
      if (err.message.includes("Failed to fetch")) {
        setMessage("✅ Успішна авторизація через Firebase, але не вдалося підключитися до бекенду.");
      } else {
        setMessage("❌ Помилка: " + err.message);
      }
    }    
  };

  return (
    <div style={{ maxWidth: 400, margin: "auto", padding: 20 }}>
      <h2>{isRegistering ? "Реєстрація" : "Вхід"}</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="email"
          placeholder="Email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          style={{ width: "100%", marginBottom: 10 }}
        />
        <input
          type="password"
          placeholder="Пароль"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={{ width: "100%", marginBottom: 10 }}
        />
        <button type="submit" style={{ width: "100%" }}>
          {isRegistering ? "Зареєструватися" : "Увійти"}
        </button>
      </form>
      <p style={{ marginTop: 10 }}>
        {isRegistering ? "Вже є акаунт?" : "Немає акаунту?"}{" "}
        <button
          onClick={() => {
            setIsRegistering(!isRegistering);
            setMessage("");
          }}
        >
          {isRegistering ? "Увійти" : "Зареєструватися"}
        </button>
      </p>
      {message && <p>{message}</p>}
    </div>
  );
};

export default LoginPage;
