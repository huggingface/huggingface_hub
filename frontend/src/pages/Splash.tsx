import React from 'react';
import { Link } from 'react-router-dom';

const Splash: React.FC = () => {
  return (
    <div>
      <h1>Welcome to PokeForge OpSec AI Agent</h1>
      <p>
        Your all-in-one tool for Pokémon TCG data, price tracking, and
        security testing.
      </p>
      <Link to="/app">
        <button>Login</button>
      </Link>
    </div>
  );
};

export default Splash;
