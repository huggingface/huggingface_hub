import * as CryptoJS from 'crypto-js';

// This would be your Cloudflare Worker environment
interface Env {
  API_KEYS: KVNamespace;
  ENCRYPTION_KEY: string;
}

export class ApiKeyManager {
  private readonly env: Env;
  private readonly encryptionKey: string;

  constructor(env: Env) {
    this.env = env;
    this.encryptionKey = env.ENCRYPTION_KEY;
  }

  private async encrypt(text: string): Promise<string> {
    return CryptoJS.AES.encrypt(text, this.encryptionKey).toString();
  }

  private async decrypt(ciphertext: string): Promise<string> {
    const bytes = CryptoJS.AES.decrypt(ciphertext, this.encryptionKey);
    return bytes.toString(CryptoJS.enc.Utf8);
  }

  private getUserServiceKey(userEmail: string, service: string): string {
    return `user:${userEmail}:service:${service}`;
  }

  public async saveKey(userEmail: string, service: string, key: string): Promise<void> {
    const encryptedKey = await this.encrypt(key);
    const storageKey = this.getUserServiceKey(userEmail, service);
    await this.env.API_KEYS.put(storageKey, encryptedKey);
  }

  public async getKey(userEmail: string, service: string): Promise<string | null> {
    const storageKey = this.getUserServiceKey(userEmail, service);
    const encryptedKey = await this.env.API_KEYS.get(storageKey);
    if (!encryptedKey) {
      return null;
    }
    return this.decrypt(encryptedKey);
  }
}
