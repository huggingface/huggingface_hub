export interface MonitoredProduct {
  id: string; // A unique ID for the product
  userEmail: string;
  url: string;
  priceThreshold: number; // Alert if the price drops below this value
  lastStatus: 'In Stock' | 'Out of Stock' | 'Unknown';
  lastPrice: number | null;
}

// This would be your Cloudflare Worker environment
interface Env {
  MONITORED_PRODUCTS: KVNamespace;
}

export class MonitoringService {
  private readonly env: Env;

  constructor(env: Env) {
    this.env = env;
  }

  // Generates a unique key for storing a user's list of products
  private getUserProductsKey(userEmail: string): string {
    return `user:${userEmail}:products`;
  }

  public async getProducts(userEmail: string): Promise<MonitoredProduct[]> {
    const productsJson = await this.env.MONITORED_PRODUCTS.get(this.getUserProductsKey(userEmail));
    return productsJson ? JSON.parse(productsJson) : [];
  }

  public async addProduct(userEmail: string, url: string, priceThreshold: number): Promise<void> {
    const products = await this.getProducts(userEmail);
    const newProduct: MonitoredProduct = {
      id: crypto.randomUUID(),
      userEmail,
      url,
      priceThreshold,
      lastStatus: 'Unknown',
      lastPrice: null,
    };
    products.push(newProduct);
    await this.env.MONITORED_PRODUCTS.put(this.getUserProductsKey(userEmail), JSON.stringify(products));
  }

  public async removeProduct(userEmail: string, productId: string): Promise<void> {
      let products = await this.getProducts(userEmail);
      products = products.filter(p => p.id !== productId);
      await this.env.MONITORED_PRODUCTS.put(this.getUserProductsKey(userEmail), JSON.stringify(products));
  }

  public async scanProducts(): Promise<void> {
    // In a real implementation, you would list all keys in the KV namespace
    // For this example, we'll assume we can get all users' products
    // This is a simplification; a real implementation would need a more robust way to iterate all users.
    console.log("Starting product scan...");
    // The logic to fetch all users and their products would go here.
    // For each product, fetch the URL, parse the content for stock status and price.
    // If a change is detected, trigger an alert.
    console.log("Product scan complete.");
  }
}
