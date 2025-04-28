import React, { useState } from "react";
import { auth } from "../firebaseConfig";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
} from "firebase/auth";
import axios from "axios";

export default function LoginRegister() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState("login");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const userCredential =
        mode === "login"
          ? await signInWithEmailAndPassword(auth, email, password)
          : await createUserWithEmailAndPassword(auth, email, password);

      const token = await userCredential.user.getIdToken();

      const res = await axios.get("http://localhost:8000/auth/profile", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      alert("Welcome, " + res.data.email);
    } catch (err) {
      console.error(err);
      alert("Auth error");
    }
  };

  return (
    <div>
      <h2>{mode === "login" ? "Login" : "Register"}</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          required
        /><br/>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          required
        /><br/>
        <button type="submit">
          {mode === "login" ? "Login" : "Register"}
        </button>
      </form>
      <button onClick={() => setMode(mode === "login" ? "register" : "login")}>
        {mode === "login" ? "Switch to Register" : "Switch to Login"}
      </button>
    </div>
  );
}
