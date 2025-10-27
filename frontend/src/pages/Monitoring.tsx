import React, { useState, useEffect } from 'react';

interface MonitoredProduct {
  id: string;
  url: string;
  priceThreshold: number;
  lastStatus: string;
  lastPrice: number | null;
}

const Monitoring: React.FC = () => {
  const [products, setProducts] = useState<MonitoredProduct[]>([]);
  const [url, setUrl] = useState('');
  const [priceThreshold, setPriceThreshold] = useState('');
  const [message, setMessage] = useState('');

  useEffect(() => {
    fetchProducts();
  }, []);

  const fetchProducts = async () => {
    try {
      const response = await fetch('/api/monitoring/products');
      if (response.ok) {
        const data = await response.json();
        setProducts(data);
      } else {
        setMessage('Failed to fetch products.');
      }
    } catch (error) {
      setMessage('An error occurred while fetching products.');
    }
  };

  const handleAddProduct = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await fetch('/api/monitoring/products', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, priceThreshold: parseFloat(priceThreshold) }),
      });
      if (response.ok) {
        setUrl('');
        setPriceThreshold('');
        fetchProducts();
      } else {
        setMessage('Failed to add product.');
      }
    } catch (error) {
      setMessage('An error occurred while adding the product.');
    }
  };

  const handleRemoveProduct = async (productId: string) => {
    try {
      const response = await fetch(`/api/monitoring/products/${productId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        fetchProducts();
      } else {
        setMessage('Failed to remove product.');
      }
    } catch (error) {
      setMessage('An error occurred while removing the product.');
    }
  };

  return (
    <div>
      <h2>Product Monitoring</h2>
      <p>Add a product URL to monitor for restocks and price drops.</p>
      <form onSubmit={handleAddProduct}>
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com/product/123"
          required
        />
        <input
          type="number"
          value={priceThreshold}
          onChange={(e) => setPriceThreshold(e.target.value)}
          placeholder="Price Threshold"
          required
        />
        <button type="submit">Add Product</button>
      </form>
      {message && <p>{message}</p>}
      <h3>Monitored Products</h3>
      <ul>
        {products.map((product) => (
          <li key={product.id}>
            {product.url} (Price Threshold: ${product.priceThreshold})
            <button onClick={() => handleRemoveProduct(product.id)}>Remove</button>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Monitoring;
