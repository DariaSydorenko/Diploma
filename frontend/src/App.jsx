import { BrowserRouter, Routes, Route } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
// import SearchPage from './pages/SearchPage';

export const App = () => (
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<LoginPage />} />
      {/* <Route path="/search" element={<SearchPage />} /> */}
    </Routes>
  </BrowserRouter>
);
