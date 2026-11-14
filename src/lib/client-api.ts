export async function readApiJson<T>(response: Response, endpointName: string): Promise<T> {
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().includes("application/json")) {
    const reason = response.status === 404
      ? "The API route was not found. Restart the development server from the project directory."
      : "The server returned an HTML page instead of API data. Check the Next.js terminal for the route error.";
    throw new Error(`${endpointName} failed (HTTP ${response.status}). ${reason}`);
  }

  return response.json() as Promise<T>;
}
