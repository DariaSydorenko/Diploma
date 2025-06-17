import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
// import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
  apiKey: "AIzaSyBdWJj4FobiDdHNv_OrYgMfZuDINGCr804",
  authDomain: "assistant-for-scientists.firebaseapp.com",
  projectId: "assistant-for-scientists",
  storageBucket: "assistant-for-scientists.firebasestorage.app",
  messagingSenderId: "153891253963",
  appId: "1:153891253963:web:5eee8d8fdd828a27b8b54e",
  measurementId: "G-VGCM5DYPKS"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
// const analytics = getAnalytics(app);