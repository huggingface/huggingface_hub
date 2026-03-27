import React, { useState } from 'react';

const Home: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any>(null);
  const [message, setMessage] = useState('');

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage('');
    setResults(null);

    try {
      const response = await fetch(`/api/search?q=${query}`);
      if (response.ok) {
        const data = await response.json();
        setResults(data);
      } else {
        const errorText = await response.text();
        setMessage(`Error: ${errorText}`);
      }
    } catch (error) {
      setMessage(`An unexpected error occurred: ${error}`);
    }
  };

  return (
    <div>
      <h2>Search for Pokémon Cards</h2>
      <form onSubmit={handleSearch}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter a card name"
          required
        />
        <button type="submit">Search</button>
      </form>
      {message && <p>{message}</p>}
      {results && (
        <div>
          <h3>Search Results</h3>
          <pre>{JSON.stringify(results, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default Home;
