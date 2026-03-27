export class GoogleSearchClient {
  private readonly apiKey: string;
  private readonly cseId: string;

  constructor(apiKey: string, cseId: string) {
    this.apiKey = apiKey;
    this.cseId = cseId;
  }

  public async search(query: string): Promise<any> {
    const url = `https://www.googleapis.com/customsearch/v1?key=${this.apiKey}&cx=${this.cseId}&q=${query}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Google Search API request failed with status ${response.status}`);
    }
    return response.json();
  }
}
