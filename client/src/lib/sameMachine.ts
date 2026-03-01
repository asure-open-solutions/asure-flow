/**
 * Determine if the server is running on the same machine as the client.
 */
export function isSameMachine(serverUrl: string, serverHostname?: string): boolean {
  try {
    const url = new URL(serverUrl);
    const host = url.hostname.toLowerCase();

    if (host === "localhost" || host === "127.0.0.1" || host === "::1") {
      return true;
    }

    // Compare URL hostname with the server's reported hostname
    if (serverHostname && host === serverHostname.toLowerCase()) {
      return true;
    }

    return false;
  } catch {
    return false;
  }
}
