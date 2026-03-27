import React from 'react';
import { Link, Outlet } from 'react-router-dom';

const MainApp: React.FC = () => {
  return (
    <div>
      <nav>
        <ul>
          <li>
            <Link to="/app">Home</Link>
          </li>
          <li>
            <Link to="/app/settings">Settings</Link>
          </li>
          <li>
            <Link to="/app/pentest">Penetration Testing</Link>
          </li>
          <li>
            <Link to="/app/monitoring">Monitoring</Link>
          </li>
        </ul>
      </nav>
      <hr />
      <Outlet />
    </div>
  );
};

export default MainApp;
