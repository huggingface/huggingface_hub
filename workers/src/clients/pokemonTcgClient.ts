export class PokemonTcgClient {
  private readonly apiKey: string;
  private readonly baseUrl = 'https://api.pokemontcg.io/v2';

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  private async fetchFromApi(endpoint: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/${endpoint}`, {
      headers: {
        'X-Api-Key': this.apiKey,
      },
    });

    if (!response.ok) {
      throw new Error(`Pokémon TCG API request failed with status ${response.status}`);
    }

    return response.json();
  }

  public getCards(query?: string): Promise<any> {
    const endpoint = `cards${query ? `?q=${encodeURIComponent(query)}` : ''}`;
    return this.fetchFromApi(endpoint);
  }

  public getSets(): Promise<any> {
    return this.fetchFromApi('sets');
  }
}
