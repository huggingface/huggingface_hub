import { verifyToken } from '../src/auth';
import { jwtVerify, createRemoteJWKSet } from 'jose';

// Mock the 'jose' library
jest.mock('jose', () => ({
  jwtVerify: jest.fn(),
  createRemoteJWKSet: jest.fn(),
}));

const mockEnv = {
  CLOUDFLARE_ACCESS_AUD: 'mock-audience',
};

describe('Auth Middleware', () => {
  beforeEach(() => {
    // Reset mocks before each test
    (jwtVerify as jest.Mock).mockClear();
    (createRemoteJWKSet as jest.Mock).mockClear();
  });

  it('should return valid and payload for a valid JWT', async () => {
    const mockPayload = { email: 'test@example.com', iat: 1234567890 };
    (jwtVerify as jest.Mock).mockResolvedValue({ payload: mockPayload });

    const mockRequest = new Request('https://example.com', {
      headers: { 'CF-Access-JWT-Assertion': 'valid.jwt.token' },
    });

    const result = await verifyToken(mockRequest, mockEnv);

    expect(result.valid).toBe(true);
    expect(result.payload).toEqual(mockPayload);
  });

  it('should return invalid for an invalid JWT', async () => {
    (jwtVerify as jest.Mock).mockRejectedValue(new Error('Invalid signature'));

    const mockRequest = new Request('https://example.com', {
      headers: { 'CF-Access-JWT-Assertion': 'invalid.jwt.token' },
    });

    const result = await verifyToken(mockRequest, mockEnv);

    expect(result.valid).toBe(false);
    expect(result.payload).toBeUndefined();
  });

  it('should return invalid if the JWT is missing', async () => {
    const mockRequest = new Request('https://example.com');

    const result = await verifyToken(mockRequest, mockEnv);

    expect(result.valid).toBe(false);
    expect(result.payload).toBeUndefined();
  });
});
