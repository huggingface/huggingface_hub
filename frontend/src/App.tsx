import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Splash from './pages/Splash';
import MainApp from './pages/MainApp';
import Settings from './pages/Settings';
import Home from './pages/Home';
import PenTest from './pages/PenTest';
import Monitoring from './pages/Monitoring';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Splash />} />
        <Route path="/app" element={<MainApp />}>
          <Route index element={<Home />} />
          <Route path="settings" element={<Settings />} />
          <Route path="pentest" element={<PenTest />} />
          <Route path="monitoring" element={<Monitoring />} />
        </Route>
      </Routes>
    </Router>
  );
};

export default App;
