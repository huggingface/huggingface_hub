import { jwtVerify, createRemoteJWKSet } from 'jose';

interface Env {
  CLOUDFLARE_ACCESS_AUD: string;
}

// The JWKS endpoint for your Cloudflare Access tenant
// You can find this in your Cloudflare Zero Trust dashboard
// under Access -> Service Auth -> Your Application
const JWKS_URL = "https://goldshore.cloudflareaccess.com/cdn-cgi/access/certs";

export async function verifyToken(request: Request, env: Env): Promise<{ valid: boolean; payload?: any }> {
  const jwt = request.headers.get('CF-Access-JWT-Assertion');
  if (!jwt) {
    return { valid: false };
  }

  const JWKS = createRemoteJWKSet(new URL(JWKS_URL));

  try {
    const { payload } = await jwtVerify(jwt, JWKS, {
      issuer: 'https://goldshore.cloudflareaccess.com',
      audience: env.CLOUDFLARE_ACCESS_AUD,
    });
    return { valid: true, payload };
  } catch (err) {
    console.error(err);
    return { valid: false };
  }
}
