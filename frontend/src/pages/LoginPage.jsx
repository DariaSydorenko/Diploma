// import React, { useState } from "react";
// import { auth } from "../firebaseConfig";
// import {
//   signInWithEmailAndPassword,
//   createUserWithEmailAndPassword,
// } from "firebase/auth";
// import { useNavigate } from 'react-router-dom';

// const LoginPage = () => {
//   const [email, setEmail] = useState("");
//   const [password, setPassword] = useState("");
//   const [isRegistering, setIsRegistering] = useState(false);
//   const [message, setMessage] = useState("");
//   const navigate = useNavigate();

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     try {
//       let userCredential;
//       if (isRegistering) {
//         userCredential = await createUserWithEmailAndPassword(auth, email, password);
//       } else {
//         userCredential = await signInWithEmailAndPassword(auth, email, password);
//       }

//       const idToken = await userCredential.user.getIdToken();

//       const response = await fetch("http://localhost:8000/auth/verify-token", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//           Authorization: `Bearer ${idToken}`,
//         },
//         body: JSON.stringify({ email }),
//       });
    
//       const data = await response.json();

//       console.log(response)
    
//       if (response.ok) {
//         navigate('/search_articles');
//         setMessage(`✅ Успішно! Привіт, ${data.name || email}`);
//       } else {
//         setMessage(`✅ Успішна авторизація через Firebase, але сервер відповів з помилкою: ${data.detail}`);
//       }
//     } catch (err) {
//       console.error(err);

//       console.error(err);

//       let userMessage = "❌ Помилка: ";
    
//       if (err.code) {
//         switch (err.code) {
//           case "auth/invalid-email":
//             userMessage += "Неправильний формат електронної пошти.";
//             break;
//           case "auth/user-not-found":
//             userMessage += "Користувача з такою поштою не знайдено.";
//             break;
//           case "auth/wrong-password":
//             userMessage += "Неправильний пароль.";
//             break;
//           case "auth/email-already-in-use":
//             userMessage += "Цей email вже зареєстрований.";
//             break;
//           case "auth/weak-password":
//             userMessage += "Пароль занадто слабкий. Мінімум 6 символів.";
//             break;
//           case "auth/missing-password":
//             userMessage += "Введіть пароль.";
//             break;
//           case "auth/network-request-failed":
//             userMessage += "Проблема з мережею. Перевірте з'єднання.";
//             break;
//           default:
//             userMessage += err.message;
//         }
//       } else if (err.message.includes("Failed to fetch")) {
//         userMessage = "✅ Успішна авторизація через Firebase, але не вдалося підключитися до бекенду.";
//       } else {
//         userMessage += err.message;
//       }
    
//       setMessage(userMessage);
//     }    
//   };

//   return (
//     <div style={{ maxWidth: 400, margin: "auto", padding: 20 }}>
//       <h2>{isRegistering ? "Реєстрація" : "Вхід"}</h2>
//       <form onSubmit={handleSubmit}>
//         <input
//           type="email"
//           placeholder="Email"
//           required
//           value={email}
//           onChange={(e) => setEmail(e.target.value)}
//           style={{ width: "100%", marginBottom: 10 }}
//         />
//         <input
//           type="password"
//           placeholder="Пароль"
//           required
//           value={password}
//           onChange={(e) => setPassword(e.target.value)}
//           style={{ width: "100%", marginBottom: 10 }}
//         />
//         <button type="submit" style={{ width: "100%" }}>
//           {isRegistering ? "Зареєструватися" : "Увійти"}
//         </button>
//       </form>
//       <p style={{ marginTop: 10 }}>
//         {isRegistering ? "Вже є акаунт?" : "Немає акаунту?"}{" "}
//         <button
//           onClick={() => {
//             setIsRegistering(!isRegistering);
//             setMessage("");
//           }}
//         >
//           {isRegistering ? "Увійти" : "Зареєструватися"}
//         </button>
//       </p>
//       {message && <p>{message}</p>}
//     </div>
//   );
// };

// export default LoginPage;
