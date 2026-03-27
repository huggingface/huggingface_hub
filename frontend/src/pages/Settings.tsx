import React, { useState } from 'react';

const Settings: React.FC = () => {
  const [service, setService] = useState('gemini');
  const [apiKey, setApiKey] = useState('');
  const [message, setMessage] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage('');

    try {
      const response = await fetch(`/api/keys/${service}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ key: apiKey }),
      });

      if (response.ok) {
        setMessage(`API key for ${service} saved successfully.`);
        setApiKey('');
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
      <h2>Settings</h2>
      <p>Manage your API keys here.</p>
      <form onSubmit={handleSubmit}>
        <label>
          Service:
          <select value={service} onChange={(e) => setService(e.target.value)}>
            <option value="gemini">Gemini</option>
            <option value="google">Google</option>
            <option value="google_cse_id">Google CSE ID</option>
            <option value="twilio_sid">Twilio SID</option>
            <option value="twilio_token">Twilio Token</option>
          </select>
        </label>
        <br />
        <label>
          API Key:
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            required
          />
        </label>
        <br />
        <button type="submit">Save Key</button>
      </form>
      {message && <p>{message}</p>}
    </div>
  );
};

export default Settings;
