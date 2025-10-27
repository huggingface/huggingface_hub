import { ApiKeyManager } from '../src/services/apiKeyManager';
import * as CryptoJS from 'crypto-js';

// Mock the 'crypto-js' library
jest.mock('crypto-js', () => ({
  AES: {
    encrypt: jest.fn().mockReturnValue({ toString: () => 'encrypted-key' }),
    decrypt: jest.fn().mockReturnValue({ toString: () => 'decrypted-key' }),
  },
  enc: {
    Utf8: {},
  },
}));

const mockKV = {
  put: jest.fn(),
  get: jest.fn(),
};

const mockEnv = {
  API_KEYS: mockKV as any,
  ENCRYPTION_KEY: 'mock-secret-key',
};

describe('ApiKeyManager', () => {
  let apiKeyManager: ApiKeyManager;

  beforeEach(() => {
    apiKeyManager = new ApiKeyManager(mockEnv);
    mockKV.put.mockClear();
    mockKV.get.mockClear();
    (CryptoJS.AES.encrypt as jest.Mock).mockClear();
    (CryptoJS.AES.decrypt as jest.Mock).mockClear();
  });

  it('should encrypt and save a key', async () => {
    await apiKeyManager.saveKey('test@example.com', 'gemini', 'my-api-key');
    expect(CryptoJS.AES.encrypt).toHaveBeenCalledWith('my-api-key', 'mock-secret-key');
    expect(mockKV.put).toHaveBeenCalledWith('user:test@example.com:service:gemini', 'encrypted-key');
  });

  it('should retrieve and decrypt a key', async () => {
    mockKV.get.mockResolvedValue('encrypted-key-from-kv');
    const key = await apiKeyManager.getKey('test@example.com', 'gemini');

    expect(mockKV.get).toHaveBeenCalledWith('user:test@example.com:service:gemini');
    expect(CryptoJS.AES.decrypt).toHaveBeenCalledWith('encrypted-key-from-kv', 'mock-secret-key');
    expect(key).toBe('decrypted-key');
  });

  it('should return null if the key is not found', async () => {
    mockKV.get.mockResolvedValue(null);
    const key = await apiKeyManager.getKey('test@example.com', 'gemini');
    expect(key).toBeNull();
  });
});
