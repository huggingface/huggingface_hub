import { ApiKeyManager } from './services/apiKeyManager';
import { GeminiClient } from './clients/geminiClient';
import { GoogleSearchClient } from './clients/googleSearchClient';
import { verifyToken } from './auth';
import { PenTestService } from './services/penTest';
import { MonitoringService } from './services/monitoringService';

export interface Env {
  API_KEYS: KVNamespace;
  MONITORED_PRODUCTS: KVNamespace;
  ENCRYPTION_KEY: string;
  CLOUDFLARE_ACCESS_AUD: string;
}

export default {
  async fetch(
    request: Request,
    env: Env,
    ctx: ExecutionContext
  ): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    if (path.startsWith('/api/')) {
      const { valid, payload } = await verifyToken(request, env);
      if (!valid || !payload || typeof payload.email !== 'string') {
        return new Response('Invalid or missing authentication token.', { status: 401 });
      }
      const userEmail = payload.email;

      const apiKeyManager = new ApiKeyManager(env);
      const monitoringService = new MonitoringService(env);

      if (method === 'PUT' && path.startsWith('/api/keys/')) {
        const service = path.split('/')[3];
        if (!request.body) return new Response('Request body is missing', { status: 400 });
        try {
          const { key } = await request.json<{ key: string }>();
          if (!service || !key) return new Response('Missing service or key', { status: 400 });
          await apiKeyManager.saveKey(userEmail, service, key);
          return new Response(`Key for ${service} saved successfully.`, { status: 200 });
        } catch (error) {
          return new Response('Invalid JSON in request body', { status: 400 });
        }
      }

      if (method === 'GET' && path === '/api/search') {
        const query = url.searchParams.get('q');
        if (!query) return new Response('Missing search query', { status: 400 });

        const googleApiKey = await apiKeyManager.getKey(userEmail, 'google');
        const googleCseId = await apiKeyManager.getKey(userEmail, 'google_cse_id');
        if (!googleApiKey || !googleCseId) return new Response('Google API key or CSE ID not configured', { status: 500 });

        const searchClient = new GoogleSearchClient(googleApiKey, googleCseId);
        const searchResults = await searchClient.search(query);
        return new Response(JSON.stringify(searchResults), { headers: { 'Content-Type': 'application/json' } });
      }

      if (method === 'POST' && path === '/api/gemini') {
        const geminiApiKey = await apiKeyManager.getKey(userEmail, 'gemini');
        if (!geminiApiKey) return new Response('Gemini API key not configured', { status: 500 });

        const geminiClient = new GeminiClient(geminiApiKey);
        if (!request.body) return new Response('Request body is missing', { status: 400 });

        const { prompt } = await request.json<{ prompt: string }>();
        if (!prompt) return new Response('Missing prompt', { status: 400 });

        const result = await geminiClient.generateText(prompt);
        return new Response(result, { status: 200 });
      }

      if (method === 'POST' && path === '/api/pentest') {
        if (!request.body) return new Response('Request body is missing', { status: 400 });
        try {
            const { targetUrl } = await request.json<{ targetUrl: string }>();
            if (!targetUrl) return new Response('Missing targetUrl', { status: 400 });

            const penTestService = new PenTestService();
            const result = await penTestService.runTest(targetUrl);
            return new Response(JSON.stringify(result), { headers: { 'Content-Type': 'application/json' } });
        } catch (error) {
            return new Response('Invalid JSON in request body', { status: 400 });
        }
      }

      if (method === 'GET' && path === '/api/monitoring/products') {
        const products = await monitoringService.getProducts(userEmail);
        return new Response(JSON.stringify(products), { headers: { 'Content-Type': 'application/json' } });
      }

      if (method === 'POST' && path === '/api/monitoring/products') {
        if (!request.body) return new Response('Request body is missing', { status: 400 });
        try {
            const { url, priceThreshold } = await request.json<{ url: string, priceThreshold: number }>();
            if (!url || !priceThreshold) return new Response('Missing url or priceThreshold', { status: 400 });
            await monitoringService.addProduct(userEmail, url, priceThreshold);
            return new Response('Product added successfully.', { status: 200 });
        } catch (error) {
            return new Response('Invalid JSON in request body', { status: 400 });
        }
      }

      if (method === 'DELETE' && path.startsWith('/api/monitoring/products/')) {
        const productId = path.split('/')[4];
        if (!productId) return new Response('Missing product ID', { status: 400 });
        await monitoringService.removeProduct(userEmail, productId);
        return new Response('Product removed successfully.', { status: 200 });
      }
    }

    return new Response('Not Found', { status: 404 });
  },

  async scheduled(
    controller: ScheduledController,
    env: Env,
    ctx: ExecutionContext
  ): Promise<void> {
    const monitoringService = new MonitoringService(env);
    ctx.waitUntil(monitoringService.scanProducts());
  },
};
