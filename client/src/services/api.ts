/**
 * REST API client for the Asuré Flow server.
 */

import type {
  Session,
  SessionSummary,
  SessionSettings,
  ServerConfig,
  LLMProviderConfig,
  UserProfile,
  AudioDeviceInfo,
  ClientAudioDevice,
  Preset,
  Participant,
  TranscriptEntry,
} from "@/types";

let baseUrl = "http://localhost:8000";

export function setServerUrl(url: string) {
  let normalized = url.replace(/\/+$/, "");
  // Auto-prepend http:// if user entered a bare host/IP
  if (normalized && !/^https?:\/\//i.test(normalized)) {
    normalized = `http://${normalized}`;
  }
  baseUrl = normalized;
}

export function getServerUrl() {
  return baseUrl;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${baseUrl}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ── Health ──

export interface HealthStatus {
  online: boolean;
  llmAvailable: boolean;
  llmProvider: string | null;
}

export async function checkHealth(): Promise<{ status: string; llm_available: boolean }> {
  return request("/api/health");
}

/** Non-throwing health check — returns server online status + LLM availability. */
export async function checkServerHealth(): Promise<HealthStatus> {
  try {
    const res = await fetch(`${baseUrl}/api/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return { online: false, llmAvailable: false, llmProvider: null };
    const data = await res.json();
    return {
      online: true,
      llmAvailable: data.llm_available ?? false,
      llmProvider: data.llm_provider ?? null,
    };
  } catch {
    return { online: false, llmAvailable: false, llmProvider: null };
  }
}

/** Non-throwing connectivity check — returns true if server responds to /api/health. */
export async function isServerReachable(): Promise<boolean> {
  return (await checkServerHealth()).online;
}

// ── Sessions ──

export async function listSessions(): Promise<SessionSummary[]> {
  return request("/api/sessions");
}

export async function createSession(name?: string): Promise<Session> {
  return request("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ name: name || "Untitled Session" }),
  });
}

export async function getSession(id: string): Promise<Session> {
  return request(`/api/sessions/${id}`);
}

export async function deleteSession(id: string): Promise<void> {
  await request(`/api/sessions/${id}`, { method: "DELETE" });
}

export async function renameSession(id: string, name: string): Promise<Session> {
  return request(`/api/sessions/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ name }),
  });
}

// ── Transcript Entry Actions ──

export async function editTranscriptEntry(
  sessionId: string,
  entryId: string,
  text: string,
): Promise<TranscriptEntry> {
  return request(`/api/sessions/${sessionId}/transcript/${entryId}`, {
    method: "PATCH",
    body: JSON.stringify({ text }),
  });
}

export async function deleteTranscriptEntry(
  sessionId: string,
  entryId: string,
): Promise<void> {
  await request(`/api/sessions/${sessionId}/transcript/${entryId}`, {
    method: "DELETE",
  });
}

// ── Session Settings (per-session overrides) ──

export async function updateSessionSettings(
  sessionId: string,
  settings: Partial<SessionSettings>,
): Promise<SessionSettings> {
  return request(`/api/sessions/${sessionId}/settings`, {
    method: "PATCH",
    body: JSON.stringify(settings),
  });
}

export async function clearSessionSettings(sessionId: string): Promise<void> {
  await request(`/api/sessions/${sessionId}/settings`, { method: "DELETE" });
}

export async function exportSession(id: string): Promise<Session> {
  return request(`/api/sessions/${id}/export`);
}

export async function exportSessionMarkdown(id: string): Promise<string> {
  const res = await fetch(`${baseUrl}/api/sessions/${id}/export/markdown`);
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.text();
}

// ── Participants ──

export async function getParticipants(sessionId: string): Promise<Participant[]> {
  return request(`/api/sessions/${sessionId}/participants`);
}

export async function updateParticipant(
  sessionId: string,
  speakerLabel: string,
  data: { display_name: string; role?: string },
): Promise<Participant> {
  return request(`/api/sessions/${sessionId}/participants/${encodeURIComponent(speakerLabel)}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

// ── Follow-up ──

export interface FollowupResult {
  subject: string;
  body: string;
  format: string;
}

export async function generateFollowup(
  sessionId: string,
  format: "email" | "message" | "summary" = "email",
): Promise<FollowupResult> {
  return request(`/api/sessions/${sessionId}/followup`, {
    method: "POST",
    body: JSON.stringify({ format }),
  });
}

// ── Config ──

export async function getServerConfig(): Promise<ServerConfig> {
  return request("/api/config");
}

export async function updateServerConfig(
  changes: Record<string, string | boolean | number | string[] | null>,
): Promise<ServerConfig> {
  return request("/api/config", {
    method: "PUT",
    body: JSON.stringify(changes),
  });
}

export async function resetServerConfig(): Promise<ServerConfig> {
  return request("/api/config/reset", { method: "POST" });
}

export async function getPresets(): Promise<Preset[]> {
  return request("/api/config/presets");
}

// ── Provider CRUD ──

export async function updateProvider(
  providerId: string,
  changes: Partial<Omit<LLMProviderConfig, "id" | "configured" | "api_key_hint">>,
): Promise<ServerConfig> {
  return request(`/api/config/providers/${encodeURIComponent(providerId)}`, {
    method: "PUT",
    body: JSON.stringify(changes),
  });
}

export async function addProvider(provider: {
  id: string;
  name: string;
  litellm_prefix?: string;
  model?: string;
  api_key?: string;
  api_base?: string;
  enabled?: boolean;
}): Promise<ServerConfig> {
  return request("/api/config/providers", {
    method: "POST",
    body: JSON.stringify(provider),
  });
}

export async function removeProvider(providerId: string): Promise<ServerConfig> {
  return request(`/api/config/providers/${encodeURIComponent(providerId)}`, {
    method: "DELETE",
  });
}

export async function reorderProviders(order: string[]): Promise<ServerConfig> {
  return request("/api/config/providers/order", {
    method: "PUT",
    body: JSON.stringify({ order }),
  });
}

// ── User Profile ──

export async function getProfile(): Promise<UserProfile> {
  return request("/api/profile");
}

export async function updateProfile(
  changes: Partial<UserProfile>,
): Promise<UserProfile> {
  return request("/api/profile", {
    method: "PUT",
    body: JSON.stringify(changes),
  });
}

// ── Search ──

export interface SearchParams {
  query: string;
  session_id?: string;
  speaker?: string;
  max_results?: number;
}

export interface SearchResultItem {
  session_id: string;
  session_name: string;
  entry_id: string;
  speaker: string;
  text: string;
  timestamp: string;
  relevance?: number;
}

export interface SearchResponse {
  results: SearchResultItem[];
  search_type: string;
}

export async function searchTranscripts(params: SearchParams): Promise<SearchResponse> {
  return request("/api/search", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

// ── Audio Devices ──

export interface AudioDevicesResponse {
  available: boolean;
  devices: AudioDeviceInfo[];
}

export async function getServerAudioDevices(): Promise<AudioDevicesResponse> {
  return request("/api/audio/devices");
}

export async function getClientAudioInputDevices(): Promise<ClientAudioDevice[]> {
  try {
    // Request permission first so browser returns device labels
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach((t) => t.stop());
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((d) => d.kind === "audioinput")
      .map((d) => ({
        deviceId: d.deviceId,
        label: d.label || `Microphone ${d.deviceId.slice(0, 8)}`,
      }));
  } catch {
    return [];
  }
}
