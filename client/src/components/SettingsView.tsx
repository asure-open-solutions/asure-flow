import { useEffect, useState } from "react";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSessionStore } from "@/stores/sessionStore";
import {
  getServerConfig,
  updateServerConfig,
  resetServerConfig,
  getProfile,
  updateProfile,
  getServerAudioDevices,
  getClientAudioInputDevices,
  getPresets,
  updateSessionSettings,
  updateProvider as apiUpdateProvider,
  addProvider as apiAddProvider,
  removeProvider as apiRemoveProvider,
  reorderProviders as apiReorderProviders,
} from "@/services/api";
import type { ServerConfig, LLMProviderConfig, UserProfile, AudioDeviceInfo, ClientAudioDevice, Preset, SessionSettings, FeatureToggles } from "@/types";
import { isSameMachine } from "@/lib/sameMachine";
import { cn } from "@/lib/utils";
import {
  Settings,
  Server,
  Brain,
  Wand2,
  Mic2,
  Cpu,
  Layers,
  Shield,
  Lock,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Loader2,
  Check,
  AlertTriangle,
  RotateCcw,
  Globe,
  Pin,
  Plus,
  Trash2,
} from "lucide-react";

// ── Toggle switch component ──

function Toggle({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      onClick={(e) => {
        e.stopPropagation();
        onChange(!checked);
      }}
      className={cn(
        "relative h-6 w-11 shrink-0 rounded-full transition-colors",
        checked ? "bg-blue-600" : "bg-white/10",
      )}
    >
      <span
        className={cn(
          "absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white transition-transform",
          checked && "translate-x-5",
        )}
      />
    </button>
  );
}

// ── Toggle card (reusable for feature/audio/overlay toggles) ──

function ToggleCard({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div
      onClick={() => onChange(!checked)}
      className={cn(
        "flex items-center justify-between rounded-lg border px-4 py-3 cursor-pointer transition-colors",
        checked
          ? "border-blue-500/30 bg-blue-500/5"
          : "border-white/10 bg-white/[0.02] hover:bg-white/5",
      )}
    >
      <div className="mr-3">
        <p className="text-sm font-medium text-white/90">{label}</p>
        <p className="text-xs text-white/40 mt-0.5">{description}</p>
      </div>
      <Toggle checked={checked} onChange={onChange} />
    </div>
  );
}

// ── Scoped toggle card (with session/global scope indicator) ──

function ScopedToggleCard({
  label,
  description,
  settingsKey,
  checked,
  onGlobalChange,
}: {
  label: string;
  description: string;
  settingsKey: keyof SessionSettings;
  checked: boolean;
  onGlobalChange: (v: boolean) => void;
}) {
  const currentSession = useSessionStore((s) => s.currentSession);
  const sessionOverrides = useSettingsStore((s) => s.sessionOverrides);
  const updateSessionOverride = useSettingsStore((s) => s.updateSessionOverride);
  const removeSessionOverride = useSettingsStore((s) => s.removeSessionOverride);

  const isSessionScoped = currentSession != null && sessionOverrides?.[settingsKey] !== undefined && sessionOverrides?.[settingsKey] !== null;

  const handleToggleScope = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!currentSession) return;

    if (isSessionScoped) {
      // Unpin from session — remove override
      removeSessionOverride(settingsKey);
      if (currentSession) {
        // Persist removal to server
        updateSessionSettings(currentSession.id, { [settingsKey]: null }).catch(() => {});
      }
    } else {
      // Pin to session — copy current value as override
      updateSessionOverride(settingsKey, checked as SessionSettings[typeof settingsKey]);
      if (currentSession) {
        updateSessionSettings(currentSession.id, { [settingsKey]: checked }).catch(() => {});
      }
    }
  };

  const handleValueChange = (v: boolean) => {
    if (isSessionScoped && currentSession) {
      // Update session override only
      updateSessionOverride(settingsKey, v as SessionSettings[typeof settingsKey]);
      updateSessionSettings(currentSession.id, { [settingsKey]: v }).catch(() => {});
    } else {
      // Update global
      onGlobalChange(v);
    }
  };

  return (
    <div
      onClick={() => handleValueChange(!checked)}
      className={cn(
        "flex items-center justify-between rounded-lg border px-4 py-3 cursor-pointer transition-colors",
        checked
          ? isSessionScoped
            ? "border-amber-500/30 bg-amber-500/5"
            : "border-blue-500/30 bg-blue-500/5"
          : "border-white/10 bg-white/[0.02] hover:bg-white/5",
      )}
    >
      <div className="mr-3 flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <p className="text-sm font-medium text-white/90">{label}</p>
          {isSessionScoped && (
            <span className="text-[10px] text-amber-400/70 bg-amber-400/10 rounded px-1 py-0.5">session</span>
          )}
        </div>
        <p className="text-xs text-white/40 mt-0.5">{description}</p>
      </div>
      <div className="flex items-center gap-2">
        {currentSession && (
          <button
            onClick={handleToggleScope}
            className={cn(
              "rounded p-1 transition-colors",
              isSessionScoped
                ? "text-amber-400 hover:bg-amber-400/10"
                : "text-white/20 hover:text-white/50 hover:bg-white/5",
            )}
            title={isSessionScoped ? "Using session setting (click for global)" : "Using global setting (click to override for this session)"}
          >
            {isSessionScoped ? <Pin className="h-3.5 w-3.5" /> : <Globe className="h-3.5 w-3.5" />}
          </button>
        )}
        <Toggle checked={checked} onChange={handleValueChange} />
      </div>
    </div>
  );
}

// ── Device dropdown ──

function DeviceDropdown({
  label,
  devices,
  selectedId,
  onChange,
  disabled = false,
}: {
  label: string;
  devices: { id: string; name: string }[];
  selectedId: string;
  onChange: (id: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="mb-3">
      <label className="text-xs text-white/50 mb-1 block">{label}</label>
      <select
        value={selectedId}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className={cn(
          "w-full rounded-md bg-white/5 border border-white/10 px-3 py-1.5 text-sm text-white",
          "focus:outline-none focus:ring-2 focus:ring-blue-500/50",
          disabled && "opacity-40 cursor-not-allowed",
        )}
      >
        <option value="">System Default</option>
        {devices.map((d) => (
          <option key={d.id} value={d.id}>
            {d.name}
          </option>
        ))}
      </select>
    </div>
  );
}

// ── Tab types ──

type SettingsTab = "llm" | "ai_tools" | "audio" | "transcription" | "privacy" | "overlay";

const TABS: { id: SettingsTab; label: string; icon: typeof Settings }[] = [
  { id: "llm", label: "LLM", icon: Brain },
  { id: "ai_tools", label: "AI", icon: Wand2 },
  { id: "audio", label: "Audio", icon: Mic2 },
  { id: "transcription", label: "Whisper", icon: Cpu },
  { id: "privacy", label: "Privacy", icon: Shield },
  ...(typeof window !== "undefined" && window.electronAPI
    ? [{ id: "overlay" as const, label: "Overlay", icon: Layers }]
    : []),
];

// Common LiteLLM prefixes for the "Add Provider" UI
const LITELLM_PREFIXES = [
  "openrouter", "openai", "anthropic", "gemini", "groq",
  "huggingface", "azure", "bedrock", "cohere", "mistral",
  "deepseek", "together_ai", "fireworks_ai", "ollama",
];

// ── Main component ──

export function SettingsView({ onClose }: { onClose: () => void }) {
  const [activeTab, setActiveTab] = useState<SettingsTab>("llm");
  const [resetting, setResetting] = useState(false);
  const resetAll = useSettingsStore((s) => s.resetAll);

  const handleResetAll = async () => {
    if (!confirm("Reset all settings to defaults? This cannot be undone.")) return;
    setResetting(true);
    try {
      await resetServerConfig();
      resetAll();
      window.electronAPI?.setContentProtection(true);
    } catch (err) {
      console.error("Failed to reset server config:", err);
      resetAll();
    } finally {
      setResetting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-2xl rounded-xl bg-zinc-900 border border-white/10 shadow-2xl flex flex-col max-h-[85vh]">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/10 px-6 py-4 shrink-0">
          <h2 className="flex items-center gap-2 text-lg font-semibold text-white">
            <Settings className="h-5 w-5" />
            Settings
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={handleResetAll}
              disabled={resetting}
              className={cn(
                "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm text-red-400/70 hover:text-red-400 hover:bg-red-500/10 transition-colors",
                resetting && "opacity-50 cursor-not-allowed",
              )}
            >
              <RotateCcw className={cn("h-3.5 w-3.5", resetting && "animate-spin")} />
              Reset
            </button>
            <button
              onClick={onClose}
              className="rounded-md px-3 py-1.5 text-sm text-white/60 hover:text-white hover:bg-white/10 transition-colors"
            >
              Done
            </button>
          </div>
        </div>

        {/* Body: sidebar tabs + content */}
        <div className="flex flex-1 min-h-0">
          {/* Tab sidebar */}
          <div className="w-40 shrink-0 border-r border-white/10 py-2">
            {TABS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={cn(
                  "flex w-full items-center gap-2 px-4 py-2 text-sm transition-colors",
                  activeTab === id
                    ? "bg-white/10 text-white font-medium"
                    : "text-white/50 hover:text-white/80 hover:bg-white/5",
                )}
              >
                <Icon className="h-4 w-4" />
                {label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-y-auto p-6">
            {activeTab === "llm" && <LLMTab />}
            {activeTab === "ai_tools" && <AIToolsTab />}
            {activeTab === "audio" && <AudioTab />}
            {activeTab === "transcription" && <TranscriptionTab />}
            {activeTab === "privacy" && <PrivacyTab />}
            {activeTab === "overlay" && <OverlayTab />}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── LLM Tab ──

function LLMTab() {
  const { serverUrl, setServerUrl } = useSettingsStore();
  const [serverConfig, setServerConfig] = useState<ServerConfig | null>(null);
  const [expandedProvider, setExpandedProvider] = useState<string | null>(null);
  const [formState, setFormState] = useState<Record<string, string>>({});
  const [savedFields, setSavedFields] = useState<Record<string, boolean>>({});
  const [showAddForm, setShowAddForm] = useState(false);
  const [newProvider, setNewProvider] = useState({ id: "", name: "", litellm_prefix: "openai", model: "" });

  useEffect(() => {
    getServerConfig()
      .then(setServerConfig)
      .catch(() => {});
  }, []);

  // Field-level form helpers (keyed by "providerId:fieldName")
  const fieldKey = (pid: string, field: string) => `${pid}:${field}`;

  const handleFieldChange = (pid: string, field: string, value: string) => {
    setFormState((prev) => ({ ...prev, [fieldKey(pid, field)]: value }));
  };

  const markSaved = (key: string) => {
    setSavedFields((prev) => ({ ...prev, [key]: true }));
    setTimeout(() => setSavedFields((prev) => { const next = { ...prev }; delete next[key]; return next; }), 2000);
  };

  const handleFieldBlur = async (pid: string, field: string) => {
    const key = fieldKey(pid, field);
    const value = formState[key];
    if (value === undefined) return;
    try {
      // Map form field names to API field names
      const apiField = field === "apiKey" ? "api_key" : field === "apiBase" ? "api_base" : field;
      const config = await apiUpdateProvider(pid, { [apiField]: value });
      setServerConfig(config);
      setFormState((prev) => { const next = { ...prev }; delete next[key]; return next; });
      markSaved(key);
    } catch (err) {
      console.error("Failed to save provider field:", err);
    }
  };

  const handleToggleProvider = async (pid: string, enabled: boolean) => {
    try {
      const config = await apiUpdateProvider(pid, { enabled });
      setServerConfig(config);
    } catch (err) {
      console.error("Failed to toggle provider:", err);
    }
  };

  const handleMoveProvider = async (pid: string, direction: "up" | "down") => {
    const providers = serverConfig?.llm_providers;
    if (!providers) return;
    const order = providers.map((p) => p.id);
    const idx = order.indexOf(pid);
    if (idx < 0) return;
    const swapIdx = direction === "up" ? idx - 1 : idx + 1;
    if (swapIdx < 0 || swapIdx >= order.length) return;
    [order[idx], order[swapIdx]] = [order[swapIdx], order[idx]];
    try {
      const config = await apiReorderProviders(order);
      setServerConfig(config);
    } catch (err) {
      console.error("Failed to reorder providers:", err);
    }
  };

  const handleDeleteProvider = async (pid: string) => {
    if (!confirm(`Remove provider "${pid}"?`)) return;
    try {
      const config = await apiRemoveProvider(pid);
      setServerConfig(config);
      if (expandedProvider === pid) setExpandedProvider(null);
    } catch (err) {
      console.error("Failed to delete provider:", err);
    }
  };

  const handleAddProvider = async () => {
    if (!newProvider.id || !newProvider.name) return;
    try {
      const config = await apiAddProvider({
        id: newProvider.id,
        name: newProvider.name,
        litellm_prefix: newProvider.litellm_prefix,
        model: newProvider.model,
      });
      setServerConfig(config);
      setNewProvider({ id: "", name: "", litellm_prefix: "openai", model: "" });
      setShowAddForm(false);
      setExpandedProvider(newProvider.id);
    } catch (err) {
      console.error("Failed to add provider:", err);
    }
  };

  const lockedSettings = serverConfig?.locked_settings ?? [];
  const isLocked = (field: string) => lockedSettings.includes(field);

  const providers: LLMProviderConfig[] = serverConfig?.llm_providers ?? [];

  return (
    <div className="space-y-5">
      {/* Server URL */}
      <div>
        <label className="flex items-center gap-2 text-sm font-medium text-white/80 mb-2">
          <Server className="h-4 w-4" />
          Server URL
        </label>
        <input
          type="url"
          value={serverUrl}
          onChange={(e) => setServerUrl(e.target.value)}
          placeholder="http://localhost:8000"
          className="w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm text-white placeholder-white/30 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
        />
        <p className="text-[11px] text-white/30 mt-1.5">Saved on this device. LLM and transcription settings are stored on the server.</p>
      </div>

      {/* Providers */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-white/80">LLM Providers</h3>
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 transition-colors"
          >
            <Plus className="h-3 w-3" />
            Add
          </button>
        </div>

        {/* Add provider form */}
        {showAddForm && (
          <div className="mb-3 rounded-lg border border-blue-500/30 bg-blue-500/5 p-4 space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-white/50 mb-1 block">ID (unique)</label>
                <input
                  type="text"
                  value={newProvider.id}
                  onChange={(e) => setNewProvider((p) => ({ ...p, id: e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, "-") }))}
                  placeholder="my-provider"
                  className="w-full rounded-md bg-white/5 border border-white/10 px-3 py-1.5 text-sm text-white placeholder-white/20 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
              </div>
              <div>
                <label className="text-xs text-white/50 mb-1 block">Display Name</label>
                <input
                  type="text"
                  value={newProvider.name}
                  onChange={(e) => setNewProvider((p) => ({ ...p, name: e.target.value }))}
                  placeholder="My Provider"
                  className="w-full rounded-md bg-white/5 border border-white/10 px-3 py-1.5 text-sm text-white placeholder-white/20 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-white/50 mb-1 block">LiteLLM Prefix</label>
                <select
                  value={newProvider.litellm_prefix}
                  onChange={(e) => setNewProvider((p) => ({ ...p, litellm_prefix: e.target.value }))}
                  className="w-full rounded-md bg-white/5 border border-white/10 px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                >
                  {LITELLM_PREFIXES.map((p) => (
                    <option key={p} value={p} className="bg-zinc-800">{p}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-white/50 mb-1 block">Model</label>
                <input
                  type="text"
                  value={newProvider.model}
                  onChange={(e) => setNewProvider((p) => ({ ...p, model: e.target.value }))}
                  placeholder="model-name"
                  className="w-full rounded-md bg-white/5 border border-white/10 px-3 py-1.5 text-sm text-white placeholder-white/20 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                />
              </div>
            </div>
            <div className="flex justify-end gap-2">
              <button onClick={() => setShowAddForm(false)} className="px-3 py-1.5 text-xs text-white/50 hover:text-white/80">Cancel</button>
              <button
                onClick={handleAddProvider}
                disabled={!newProvider.id || !newProvider.name}
                className={cn(
                  "px-3 py-1.5 text-xs rounded-md bg-blue-500/20 text-blue-400 hover:bg-blue-500/30",
                  (!newProvider.id || !newProvider.name) && "opacity-40 cursor-not-allowed",
                )}
              >
                Add Provider
              </button>
            </div>
          </div>
        )}

        <div className="space-y-2">
          {providers.map((provider, idx) => {
            const isExpanded = expandedProvider === provider.id;
            const isConfigured = provider.configured;
            const isEnabled = provider.enabled;

            const statusColor =
              isConfigured && isEnabled
                ? "bg-emerald-400"
                : isConfigured && !isEnabled
                  ? "bg-amber-400"
                  : "bg-white/20";

            return (
              <div
                key={provider.id}
                className="rounded-lg border border-white/10 overflow-hidden"
              >
                {/* Provider header */}
                <div className="flex items-center px-4 py-3 hover:bg-white/5 transition-colors">
                  {/* Reorder arrows */}
                  <div className="flex flex-col items-center mr-2 -my-1">
                    <button
                      onClick={() => handleMoveProvider(provider.id, "up")}
                      disabled={idx === 0 || isLocked("providers")}
                      className={cn(
                        "p-0.5 rounded hover:bg-white/10",
                        (idx === 0 || isLocked("providers")) && "opacity-20 pointer-events-none",
                      )}
                    >
                      <ChevronUp className="h-3 w-3 text-white/50" />
                    </button>
                    <button
                      onClick={() => handleMoveProvider(provider.id, "down")}
                      disabled={idx === providers.length - 1 || isLocked("providers")}
                      className={cn(
                        "p-0.5 rounded hover:bg-white/10",
                        (idx === providers.length - 1 || isLocked("providers")) && "opacity-20 pointer-events-none",
                      )}
                    >
                      <ChevronDown className="h-3 w-3 text-white/50" />
                    </button>
                  </div>

                  {/* Expand button */}
                  <button
                    onClick={() =>
                      setExpandedProvider(isExpanded ? null : provider.id)
                    }
                    className="flex flex-1 items-center gap-2 min-w-0"
                  >
                    <span className={cn("h-2 w-2 rounded-full shrink-0", statusColor)} />
                    <span className="text-sm font-medium text-white/90 truncate">
                      {provider.name}
                    </span>
                    <span className="text-[10px] text-white/30 truncate">
                      {provider.litellm_prefix}
                    </span>
                  </button>

                  {/* Enable/disable toggle + expand */}
                  <div className="flex items-center gap-2 ml-2">
                    {isConfigured && (
                      isLocked("providers") ? (
                        <Lock className="h-3.5 w-3.5 text-white/25" title="Managed by server admin" />
                      ) : (
                        <Toggle
                          checked={isEnabled}
                          onChange={(v) => handleToggleProvider(provider.id, v)}
                        />
                      )
                    )}
                    <button
                      onClick={() =>
                        setExpandedProvider(isExpanded ? null : provider.id)
                      }
                    >
                      {isExpanded ? (
                        <ChevronDown className="h-4 w-4 text-white/40" />
                      ) : (
                        <ChevronRight className="h-4 w-4 text-white/40" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Provider fields (expanded) */}
                {isExpanded && (
                  <div className="px-4 pb-4 space-y-3 border-t border-white/5">
                    {/* API Key */}
                    <div className="pt-3">
                      <label className="text-xs text-white/50 mb-1 flex items-center gap-1.5">
                        API Key
                        {savedFields[fieldKey(provider.id, "apiKey")] && <Check className="h-3 w-3 text-emerald-400" />}
                      </label>
                      <input
                        type="password"
                        value={formState[fieldKey(provider.id, "apiKey")] ?? ""}
                        onChange={(e) => handleFieldChange(provider.id, "apiKey", e.target.value)}
                        onBlur={() => handleFieldBlur(provider.id, "apiKey")}
                        placeholder={provider.api_key_hint || "Enter API key"}
                        className="w-full rounded-md bg-white/5 border border-white/10 px-3 py-1.5 text-sm text-white placeholder-white/20 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                      />
                    </div>

                    {/* Model */}
                    <div>
                      <label className="text-xs text-white/50 mb-1 flex items-center gap-1.5">
                        Model
                        {savedFields[fieldKey(provider.id, "model")] && <Check className="h-3 w-3 text-emerald-400" />}
                      </label>
                      <input
                        type="text"
                        value={formState[fieldKey(provider.id, "model")] ?? provider.model}
                        onChange={(e) => handleFieldChange(provider.id, "model", e.target.value)}
                        onBlur={() => handleFieldBlur(provider.id, "model")}
                        placeholder="Model name"
                        className="w-full rounded-md bg-white/5 border border-white/10 px-3 py-1.5 text-sm text-white placeholder-white/20 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                      />
                    </div>

                    {/* API Base URL */}
                    <div>
                      <label className="text-xs text-white/50 mb-1 flex items-center gap-1.5">
                        API Base URL
                        {savedFields[fieldKey(provider.id, "apiBase")] && <Check className="h-3 w-3 text-emerald-400" />}
                      </label>
                      <input
                        type="url"
                        value={formState[fieldKey(provider.id, "apiBase")] ?? provider.api_base}
                        onChange={(e) => handleFieldChange(provider.id, "apiBase", e.target.value)}
                        onBlur={() => handleFieldBlur(provider.id, "apiBase")}
                        placeholder="Default (leave empty for standard API)"
                        className="w-full rounded-md bg-white/5 border border-white/10 px-3 py-1.5 text-sm text-white placeholder-white/20 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                      />
                    </div>

                    {/* Delete button */}
                    {!isLocked("providers") && (
                      <button
                        onClick={() => handleDeleteProvider(provider.id)}
                        className="flex items-center gap-1.5 text-xs text-red-400/60 hover:text-red-400 transition-colors mt-2"
                      >
                        <Trash2 className="h-3 w-3" />
                        Remove provider
                      </button>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
}

// ── AI Tab (presets + features + tools) ──

const WHISPER_MODELS = [
  { value: "tiny", label: "Tiny", desc: "Fastest, least accurate" },
  { value: "base", label: "Base", desc: "Fast, basic accuracy" },
  { value: "small", label: "Small", desc: "Balanced speed/accuracy" },
  { value: "medium", label: "Medium", desc: "Good accuracy, slower" },
  { value: "large-v3", label: "Large v3", desc: "Best accuracy, slowest" },
  { value: "large-v3-turbo", label: "Large v3 Turbo", desc: "Near-best accuracy, faster" },
];

function AIToolsTab() {
  const { featureToggles, setFeatureToggles, setAiPreset, setCustomSystemPrompt, getEffectiveToggles } = useSettingsStore();
  const effectiveToggles = useSettingsStore((s) => s.getEffectiveToggles());
  const [presets, setPresets] = useState<Preset[]>([]);
  const [serverProfile, setServerProfile] = useState<UserProfile | null>(null);
  const [customPrompt, setCustomPrompt] = useState("");
  const [showCustomPrompt, setShowCustomPrompt] = useState(false);
  const [savingPrompt, setSavingPrompt] = useState(false);

  useEffect(() => {
    getPresets().then(setPresets).catch(() => {});
    getProfile().then((p) => {
      setServerProfile(p);
      if (p.custom_system_prompt) {
        setCustomPrompt(p.custom_system_prompt);
        setShowCustomPrompt(true);
      }
    }).catch(() => {});
  }, []);

  const handlePresetSelect = async (presetId: string) => {
    try {
      const updated = await updateProfile({ ai_preset: presetId, custom_system_prompt: null });
      setServerProfile(updated);
      setAiPreset(presetId);
      setCustomSystemPrompt(null);
      setShowCustomPrompt(false);
      setCustomPrompt("");
      // Apply preset's default tools (only boolean toggles — exclude string fields)
      const preset = presets.find((p) => p.id === presetId);
      if (preset) {
        const { deep_think, agent_mode, ...boolToggles } = preset.default_tools as Record<string, unknown>;
        setFeatureToggles(boolToggles as Partial<FeatureToggles>);
      }
    } catch (err) {
      console.error("Failed to select preset:", err);
    }
  };

  const handleSaveCustomPrompt = async () => {
    setSavingPrompt(true);
    try {
      const updated = await updateProfile({ custom_system_prompt: customPrompt });
      setServerProfile(updated);
      setCustomSystemPrompt(customPrompt);
    } catch (err) {
      console.error("Failed to save custom prompt:", err);
    } finally {
      setSavingPrompt(false);
    }
  };

  const handleClearCustomPrompt = async () => {
    try {
      const updated = await updateProfile({ custom_system_prompt: null });
      setServerProfile(updated);
      setCustomSystemPrompt(null);
      setCustomPrompt("");
      setShowCustomPrompt(false);
    } catch (err) {
      console.error("Failed to clear custom prompt:", err);
    }
  };

  const features = [
    { key: "fact_checking" as const, label: "Fact Checking", desc: "Tag claims as supported, contradicted, or uncertain" },
    { key: "suggestions" as const, label: "Response Suggestions", desc: "Generate reply suggestions in real time" },
    { key: "notes" as const, label: "Rolling Notes", desc: "Extract action items, decisions, facts, and risks" },
  ];

  const tools = [
    { key: "search_transcript" as const, label: "Transcript Search", desc: "Let AI search the current conversation for specific info" },
    { key: "search_sessions" as const, label: "Session History Search", desc: "Let AI search across all past sessions" },
    { key: "web_search" as const, label: "Web Search", desc: "Let AI search the web for fact-checking or research" },
    { key: "format_code" as const, label: "Code Analysis", desc: "Detect and analyse code in conversations (coding interviews)" },
  ];

  const deepThinkModes = [
    { value: "off" as const, label: "Off" },
    { value: "auto" as const, label: "Auto" },
    { value: "always" as const, label: "Always" },
  ];

  return (
    <div className="space-y-6">
      {/* Situation Preset */}
      <div>
        <h3 className="text-sm font-medium text-white/80 mb-3">Situation Preset</h3>
        <div className="grid grid-cols-2 gap-2">
          {presets.map((preset) => (
            <button
              key={preset.id}
              onClick={() => handlePresetSelect(preset.id)}
              className={cn(
                "rounded-lg border px-3 py-2.5 text-left transition-colors",
                serverProfile?.ai_preset === preset.id && !showCustomPrompt
                  ? "border-blue-500/30 bg-blue-500/10"
                  : "border-white/10 bg-white/[0.02] hover:bg-white/5",
              )}
            >
              <p className="text-sm font-medium text-white/90">{preset.name}</p>
              <p className="text-xs text-white/40 mt-0.5">{preset.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Custom System Prompt */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-white/80">Custom System Prompt</h3>
          <Toggle checked={showCustomPrompt} onChange={(v) => {
            setShowCustomPrompt(v);
            if (!v) handleClearCustomPrompt();
          }} />
        </div>
        {showCustomPrompt && (
          <div className="space-y-2">
            <textarea
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
              placeholder="Enter a custom system prompt to control AI behaviour..."
              rows={6}
              maxLength={4000}
              className="w-full rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-sm text-white placeholder-white/20 focus:outline-none focus:ring-2 focus:ring-blue-500/50 resize-y"
            />
            <div className="flex items-center justify-between">
              <span className="text-xs text-white/30">{customPrompt.length}/4000</span>
              <div className="flex gap-2">
                <button
                  onClick={handleClearCustomPrompt}
                  className="rounded-md px-3 py-1.5 text-xs text-white/50 hover:text-white hover:bg-white/10 transition-colors"
                >
                  Reset to preset
                </button>
                <button
                  onClick={handleSaveCustomPrompt}
                  disabled={savingPrompt}
                  className={cn(
                    "rounded-md px-3 py-1.5 text-xs font-medium bg-blue-600 text-white hover:bg-blue-500 transition-colors",
                    savingPrompt && "opacity-60 cursor-not-allowed",
                  )}
                >
                  {savingPrompt ? "Saving..." : "Save Prompt"}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* AI Features */}
      <div>
        <h3 className="text-sm font-medium text-white/80 mb-3">AI Features</h3>
        <div className="space-y-2">
          {features.map(({ key, label, desc }) => (
            <ScopedToggleCard
              key={key}
              label={label}
              description={desc}
              settingsKey={key}
              checked={effectiveToggles[key]}
              onGlobalChange={(v) => {
                setFeatureToggles({ [key]: v });
                updateProfile({ [key]: v }).catch(() => {});
              }}
            />
          ))}
        </div>
      </div>

      {/* AI Tools */}
      <div>
        <h3 className="text-sm font-medium text-white/80 mb-3">AI Tools</h3>
        <div className="space-y-2">
          {tools.map(({ key, label, desc }) => (
            <ScopedToggleCard
              key={key}
              label={label}
              description={desc}
              settingsKey={key}
              checked={effectiveToggles[key]}
              onGlobalChange={(v) => {
                setFeatureToggles({ [key]: v });
                updateProfile({ [key]: v }).catch(() => {});
              }}
            />
          ))}
        </div>
      </div>

      {/* Deep Think */}
      <div>
        <h3 className="text-sm font-medium text-white/80 mb-2">Deep Think</h3>
        <p className="text-xs text-white/40 mb-3">
          Enables step-by-step reasoning for complex situations. Auto lets the AI decide when to think deeper.
        </p>
        <DeepThinkSelector />
      </div>

      {/* Agent Mode */}
      <div>
        <h3 className="text-sm font-medium text-white/80 mb-2">Agent Mode</h3>
        <p className="text-xs text-white/40 mb-3">
          Controls how the AI processes conversation. Unified uses a single agent; Specialists runs focused micro-agents in parallel.
        </p>
        <AgentModeSelector />
      </div>
    </div>
  );
}

function DeepThinkSelector() {
  const { featureToggles, setFeatureToggles } = useSettingsStore();
  const effectiveToggles = useSettingsStore((s) => s.getEffectiveToggles());
  const currentSession = useSessionStore((s) => s.currentSession);
  const sessionOverrides = useSettingsStore((s) => s.sessionOverrides);
  const updateSessionOverride = useSettingsStore((s) => s.updateSessionOverride);
  const removeSessionOverride = useSettingsStore((s) => s.removeSessionOverride);

  const isSessionScoped = currentSession != null && sessionOverrides?.deep_think !== undefined && sessionOverrides?.deep_think !== null;

  const handleModeChange = (value: "off" | "auto" | "always") => {
    if (isSessionScoped && currentSession) {
      updateSessionOverride("deep_think", value);
      updateSessionSettings(currentSession.id, { deep_think: value }).catch(() => {});
    } else {
      setFeatureToggles({ deep_think: value });
      updateProfile({ deep_think: value }).catch(() => {});
    }
  };

  const handleToggleScope = () => {
    if (!currentSession) return;
    if (isSessionScoped) {
      removeSessionOverride("deep_think");
      updateSessionSettings(currentSession.id, { deep_think: null }).catch(() => {});
    } else {
      updateSessionOverride("deep_think", effectiveToggles.deep_think);
      updateSessionSettings(currentSession.id, { deep_think: effectiveToggles.deep_think }).catch(() => {});
    }
  };

  const deepThinkModes = [
    { value: "off" as const, label: "Off" },
    { value: "auto" as const, label: "Auto" },
    { value: "always" as const, label: "Always" },
  ];

  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <div className="flex gap-2 flex-1">
          {deepThinkModes.map(({ value, label }) => (
            <button
              key={value}
              onClick={() => handleModeChange(value)}
              className={cn(
                "flex-1 rounded-lg border px-3 py-2 text-sm font-medium transition-colors",
                effectiveToggles.deep_think === value
                  ? isSessionScoped
                    ? "border-amber-500/30 bg-amber-500/10 text-amber-400"
                    : "border-blue-500/30 bg-blue-500/10 text-blue-400"
                  : "border-white/10 bg-white/[0.02] text-white/50 hover:bg-white/5",
              )}
            >
              {label}
            </button>
          ))}
        </div>
        {currentSession && (
          <button
            onClick={handleToggleScope}
            className={cn(
              "rounded p-1 transition-colors",
              isSessionScoped
                ? "text-amber-400 hover:bg-amber-400/10"
                : "text-white/20 hover:text-white/50 hover:bg-white/5",
            )}
            title={isSessionScoped ? "Using session setting (click for global)" : "Using global setting (click to override for this session)"}
          >
            {isSessionScoped ? <Pin className="h-3.5 w-3.5" /> : <Globe className="h-3.5 w-3.5" />}
          </button>
        )}
      </div>
      {isSessionScoped && (
        <p className="text-[10px] text-amber-400/70">Overridden for this session</p>
      )}
    </div>
  );
}

function AgentModeSelector() {
  const { featureToggles, setFeatureToggles } = useSettingsStore();
  const effectiveToggles = useSettingsStore((s) => s.getEffectiveToggles());
  const currentSession = useSessionStore((s) => s.currentSession);
  const sessionOverrides = useSettingsStore((s) => s.sessionOverrides);
  const updateSessionOverride = useSettingsStore((s) => s.updateSessionOverride);
  const removeSessionOverride = useSettingsStore((s) => s.removeSessionOverride);

  const isModeSessionScoped = currentSession != null && sessionOverrides?.agent_mode !== undefined && sessionOverrides?.agent_mode !== null;
  const isParallelSessionScoped = currentSession != null && sessionOverrides?.parallel_tools !== undefined && sessionOverrides?.parallel_tools !== null;

  const handleModeChange = (value: "unified" | "specialists") => {
    if (isModeSessionScoped && currentSession) {
      updateSessionOverride("agent_mode", value);
      updateSessionSettings(currentSession.id, { agent_mode: value }).catch(() => {});
    } else {
      setFeatureToggles({ agent_mode: value });
      updateProfile({ agent_mode: value }).catch(() => {});
    }
  };

  const handleParallelToggle = (value: boolean) => {
    if (isParallelSessionScoped && currentSession) {
      updateSessionOverride("parallel_tools", value);
      updateSessionSettings(currentSession.id, { parallel_tools: value }).catch(() => {});
    } else {
      setFeatureToggles({ parallel_tools: value });
      updateProfile({ parallel_tools: value }).catch(() => {});
    }
  };

  const handleToggleModeScope = () => {
    if (!currentSession) return;
    if (isModeSessionScoped) {
      removeSessionOverride("agent_mode");
      updateSessionSettings(currentSession.id, { agent_mode: null }).catch(() => {});
    } else {
      updateSessionOverride("agent_mode", effectiveToggles.agent_mode);
      updateSessionSettings(currentSession.id, { agent_mode: effectiveToggles.agent_mode }).catch(() => {});
    }
  };

  const modes = [
    { value: "unified" as const, label: "Unified" },
    { value: "specialists" as const, label: "Specialists" },
  ];

  return (
    <div className="space-y-3">
      <div>
        <div className="flex items-center gap-2 mb-2">
          <div className="flex gap-2 flex-1">
            {modes.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => handleModeChange(value)}
                className={cn(
                  "flex-1 rounded-lg border px-3 py-2 text-sm font-medium transition-colors",
                  effectiveToggles.agent_mode === value
                    ? isModeSessionScoped
                      ? "border-amber-500/30 bg-amber-500/10 text-amber-400"
                      : "border-blue-500/30 bg-blue-500/10 text-blue-400"
                    : "border-white/10 bg-white/[0.02] text-white/50 hover:bg-white/5",
                )}
              >
                {label}
              </button>
            ))}
          </div>
          {currentSession && (
            <button
              onClick={handleToggleModeScope}
              className={cn(
                "rounded p-1 transition-colors",
                isModeSessionScoped
                  ? "text-amber-400 hover:bg-amber-400/10"
                  : "text-white/20 hover:text-white/50 hover:bg-white/5",
              )}
              title={isModeSessionScoped ? "Using session setting (click for global)" : "Using global setting (click to override for this session)"}
            >
              {isModeSessionScoped ? <Pin className="h-3.5 w-3.5" /> : <Globe className="h-3.5 w-3.5" />}
            </button>
          )}
        </div>
        {isModeSessionScoped && (
          <p className="text-[10px] text-amber-400/70">Overridden for this session</p>
        )}
      </div>

      <ToggleCard
        label="Parallel Tool Execution"
        description="Execute independent tool calls concurrently for lower latency"
        checked={effectiveToggles.parallel_tools}
        onChange={handleParallelToggle}
      />
    </div>
  );
}

// ── Audio Tab ──

function AudioTab() {
  const { audioToggles, setAudioToggles, diarization, setDiarization, serverUrl,
          micDeviceId, setMicDeviceId, systemDeviceId, setSystemDeviceId } =
    useSettingsStore();
  const effectiveDiarization = useSettingsStore((s) => s.sessionOverrides?.diarization ?? s.diarization);
  const [serverConfig, setServerConfig] = useState<ServerConfig | null>(null);
  const [clientDevices, setClientDevices] = useState<ClientAudioDevice[]>([]);
  const [serverDevices, setServerDevices] = useState<AudioDeviceInfo[]>([]);
  const [serverAudioAvailable, setServerAudioAvailable] = useState(false);

  const sameMachine = isSameMachine(serverUrl, serverConfig?.hostname);
  const captureSource = serverConfig?.audio_capture_source ?? "client";

  useEffect(() => {
    getServerConfig().then(setServerConfig).catch(() => {});
    getServerAudioDevices()
      .then((res) => {
        setServerAudioAvailable(res.available);
        setServerDevices(res.devices);
      })
      .catch(() => {});
    getClientAudioInputDevices().then(setClientDevices).catch(() => {});
  }, []);

  const handleCaptureSourceChange = async (source: "client" | "server") => {
    try {
      const config = await updateServerConfig({ audio_capture_source: source });
      setServerConfig(config);
    } catch (err) {
      console.error("Failed to update capture source:", err);
    }
  };

  const handleMicDeviceChange = async (deviceId: string) => {
    if (captureSource === "client") {
      // Client-local preference — stays in localStorage, not sent to server
      setMicDeviceId(deviceId || null);
    } else {
      try {
        const config = await updateServerConfig({ mic_device_id: deviceId || null });
        setServerConfig(config);
      } catch (err) {
        console.error("Failed to update mic device:", err);
      }
    }
  };

  const handleSystemDeviceChange = async (deviceId: string) => {
    // System audio loopback is always server-side capture
    if (captureSource === "client") {
      setSystemDeviceId(deviceId || null);
    } else {
      try {
        const config = await updateServerConfig({ system_device_id: deviceId || null });
        setServerConfig(config);
      } catch (err) {
        console.error("Failed to update system device:", err);
      }
    }
  };

  return (
    <div className="space-y-5">
      {/* Stream toggles */}
      <div className="space-y-2">
        <ToggleCard
          label="Microphone"
          description="Capture your microphone input"
          checked={audioToggles.mic}
          onChange={(v) => setAudioToggles({ mic: v })}
        />
        <ToggleCard
          label="System Audio"
          description="Capture system/desktop audio output (loopback)"
          checked={audioToggles.system}
          onChange={(v) => setAudioToggles({ system: v })}
        />
        <ScopedToggleCard
          label="Speaker Diarization"
          description="Distinguish multiple speakers on the system audio stream (requires pyannote + HuggingFace token)"
          settingsKey="diarization"
          checked={effectiveDiarization}
          onGlobalChange={async (v) => {
            setDiarization(v);
            try {
              await updateProfile({ diarization_enabled: v });
            } catch (err) {
              console.error("Failed to update diarization:", err);
            }
          }}
        />
        {diarization && (
          <div className="ml-4 space-y-2">
            <label className="block text-xs text-white/50">HuggingFace Token (for pyannote)</label>
            <input
              type="password"
              placeholder={serverConfig?.hf_diarization_token_hint || "hf_..."}
              className="w-full rounded-md border border-white/10 bg-white/[0.03] px-3 py-1.5 text-sm text-white placeholder:text-white/30 focus:border-blue-500/30 focus:outline-none"
              onBlur={async (e) => {
                const val = e.target.value.trim();
                if (val) {
                  try {
                    const config = await updateServerConfig({ hf_diarization_token: val });
                    setServerConfig(config);
                    e.target.value = "";
                  } catch (err) {
                    console.error("Failed to save HF token:", err);
                  }
                }
              }}
            />
            {serverConfig?.hf_diarization_token_hint && (
              <p className="text-xs text-white/30">
                Current: {serverConfig.hf_diarization_token_hint}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Capture source — only show if NOT same machine and server audio available */}
      {!sameMachine && serverAudioAvailable && (
        <div>
          <h3 className="text-sm font-medium text-white/80 mb-3">Microphone Capture Source</h3>
          <div className="flex gap-2">
            <button
              onClick={() =>
                captureSource !== "client" && handleCaptureSourceChange("client")
              }
              className={cn(
                "flex-1 flex items-center justify-center gap-2 rounded-lg border px-4 py-3 text-sm font-medium transition-colors",
                captureSource === "client"
                  ? "border-blue-500/30 bg-blue-500/10 text-blue-400"
                  : "border-white/10 bg-white/[0.02] text-white/50 hover:bg-white/5 cursor-pointer",
              )}
            >
              This Machine (Client)
            </button>
            <button
              onClick={() =>
                captureSource !== "server" && handleCaptureSourceChange("server")
              }
              className={cn(
                "flex-1 flex items-center justify-center gap-2 rounded-lg border px-4 py-3 text-sm font-medium transition-colors",
                captureSource === "server"
                  ? "border-blue-500/30 bg-blue-500/10 text-blue-400"
                  : "border-white/10 bg-white/[0.02] text-white/50 hover:bg-white/5 cursor-pointer",
              )}
            >
              Server Machine
            </button>
          </div>
        </div>
      )}

      {/* Device selection */}
      {sameMachine ? (
        /* Same machine: unified device list */
        (clientDevices.length > 0 || serverDevices.length > 0) && (
          <div>
            <h3 className="text-sm font-medium text-white/80 mb-3">Audio Devices</h3>
            <DeviceDropdown
              label="Microphone Input"
              devices={
                clientDevices.length > 0
                  ? clientDevices.map((d) => ({ id: d.deviceId, name: d.label }))
                  : serverDevices
                      .filter((d) => d.is_input)
                      .map((d) => ({ id: String(d.id), name: d.name }))
              }
              selectedId={captureSource === "client" ? (micDeviceId ?? "") : (serverConfig?.mic_device_id ?? "")}
              onChange={handleMicDeviceChange}
            />
            {serverAudioAvailable && (
              <DeviceDropdown
                label="System Audio Output (Loopback)"
                devices={serverDevices
                  .filter((d) => d.is_output)
                  .map((d) => ({
                    id: String(d.id),
                    name: d.name,
                  }))}
                selectedId={captureSource === "client" ? (systemDeviceId ?? "") : (serverConfig?.system_device_id ?? "")}
                onChange={handleSystemDeviceChange}
              />
            )}
          </div>
        )
      ) : (
        /* Different machines: separate client/server sections */
        <>
          {clientDevices.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-white/80 mb-3">
                Client Devices
              </h3>
              <DeviceDropdown
                label="Microphone Input"
                devices={clientDevices.map((d) => ({
                  id: d.deviceId,
                  name: d.label,
                }))}
                selectedId={captureSource === "client" ? (micDeviceId ?? "") : ""}
                onChange={handleMicDeviceChange}
                disabled={captureSource !== "client"}
              />
            </div>
          )}
          {serverAudioAvailable && (
            <div>
              <h3 className="text-sm font-medium text-white/80 mb-3">
                Server Devices
              </h3>
              <DeviceDropdown
                label="Microphone Input"
                devices={serverDevices
                  .filter((d) => d.is_input)
                  .map((d) => ({ id: String(d.id), name: d.name }))}
                selectedId={captureSource === "server" ? (serverConfig?.mic_device_id ?? "") : ""}
                onChange={handleMicDeviceChange}
                disabled={captureSource !== "server"}
              />
              <DeviceDropdown
                label="System Audio Output (Loopback)"
                devices={serverDevices
                  .filter((d) => d.is_output)
                  .map((d) => ({
                    id: String(d.id),
                    name: d.name,
                  }))}
                selectedId={serverConfig?.system_device_id ?? ""}
                onChange={handleSystemDeviceChange}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── Transcription Tab ──

function TranscriptionTab() {
  const [serverConfig, setServerConfig] = useState<ServerConfig | null>(null);
  const [switching, setSwitching] = useState(false);

  const lockedSettings = serverConfig?.locked_settings ?? [];
  const isLocked = (field: string) => lockedSettings.includes(field);

  useEffect(() => {
    getServerConfig()
      .then(setServerConfig)
      .catch(() => {});
  }, []);

  const currentDevice = serverConfig?.whisper_device ?? "cpu";
  const targetDevice = currentDevice === "cuda" ? "cpu" : "cuda";

  const handleSwitch = async () => {
    setSwitching(true);
    try {
      const config = await updateServerConfig({ whisper_device: targetDevice });
      setServerConfig(config);
    } catch (err) {
      console.error("Failed to switch device:", err);
    } finally {
      setSwitching(false);
    }
  };

  const handleModelChange = async (model: string) => {
    setSwitching(true);
    try {
      const config = await updateServerConfig({ whisper_model: model });
      setServerConfig(config);
    } catch (err) {
      console.error("Failed to change whisper model:", err);
    } finally {
      setSwitching(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* Model selector */}
      <div>
        <h3 className="text-sm font-medium text-white/80 mb-2 flex items-center gap-2">
          Whisper Model
          {isLocked("whisper_model") && <Lock className="h-3.5 w-3.5 text-white/25" title="Managed by server admin" />}
        </h3>
        <select
          value={serverConfig?.whisper_model ?? ""}
          onChange={(e) => handleModelChange(e.target.value)}
          disabled={switching || !serverConfig || isLocked("whisper_model")}
          title={isLocked("whisper_model") ? "Managed by server admin" : undefined}
          className={cn(
            "w-full rounded-md bg-white/5 border border-white/10 px-3 py-2 text-sm text-white",
            "focus:outline-none focus:ring-2 focus:ring-blue-500/50",
            (switching || isLocked("whisper_model")) && "opacity-60 cursor-not-allowed",
          )}
        >
          {WHISPER_MODELS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label} — {m.desc}
            </option>
          ))}
        </select>
      </div>

      {/* Device selector */}
      <div>
        <h3 className="text-sm font-medium text-white/80 mb-3 flex items-center gap-2">
          Compute Device
          {isLocked("whisper_device") && <Lock className="h-3.5 w-3.5 text-white/25" title="Managed by server admin" />}
        </h3>
        <div className="flex gap-2">
          <button
            onClick={currentDevice !== "cpu" && !isLocked("whisper_device") ? handleSwitch : undefined}
            disabled={switching || isLocked("whisper_device")}
            title={isLocked("whisper_device") ? "Managed by server admin" : undefined}
            className={cn(
              "flex-1 flex items-center justify-center gap-2 rounded-lg border px-4 py-3 text-sm font-medium transition-colors",
              currentDevice === "cpu"
                ? "border-blue-500/30 bg-blue-500/10 text-blue-400"
                : "border-white/10 bg-white/[0.02] text-white/50 hover:bg-white/5 cursor-pointer",
              (switching || isLocked("whisper_device")) && "opacity-60 cursor-not-allowed",
            )}
          >
            <Cpu className="h-4 w-4" />
            CPU
          </button>
          <button
            onClick={currentDevice !== "cuda" && !isLocked("whisper_device") ? handleSwitch : undefined}
            disabled={switching || isLocked("whisper_device")}
            title={isLocked("whisper_device") ? "Managed by server admin" : undefined}
            className={cn(
              "flex-1 flex items-center justify-center gap-2 rounded-lg border px-4 py-3 text-sm font-medium transition-colors",
              currentDevice === "cuda"
                ? "border-blue-500/30 bg-blue-500/10 text-blue-400"
                : "border-white/10 bg-white/[0.02] text-white/50 hover:bg-white/5 cursor-pointer",
              (switching || isLocked("whisper_device")) && "opacity-60 cursor-not-allowed",
            )}
          >
            <Cpu className="h-4 w-4" />
            CUDA (GPU)
          </button>
        </div>
        {switching && (
          <div className="flex items-center gap-2 mt-3 text-xs text-amber-400">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Reloading whisper model\u2026
          </div>
        )}
      </div>

      {/* Compute type info */}
      <div>
        <h3 className="text-sm font-medium text-white/80 mb-2">Compute Type</h3>
        <p className="text-sm text-white/60">
          {serverConfig?.whisper_compute_type ?? "\u2014"}
        </p>
      </div>

      {/* Warning */}
      <div className="flex items-start gap-2 rounded-lg border border-amber-500/20 bg-amber-500/5 px-4 py-3">
        <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
        <p className="text-xs text-amber-200/70">
          Changing the model or compute device reloads whisper. Transcription
          will pause during the reload. New models may need to download first.
        </p>
      </div>
    </div>
  );
}

// ── Privacy Tab ──

function PrivacyTab() {
  const { setPiiRedaction, setPrivacyMode } = useSettingsStore();

  const effectivePrivacy = useSettingsStore((s) => s.getEffectivePrivacyMode());
  const effectivePii = useSettingsStore((s) => s.getEffectivePiiRedaction());

  return (
    <div className="space-y-5">
      {/* Privacy Mode — master toggle */}
      <div>
        <h3 className="text-sm font-medium text-white/80 mb-3">Privacy Mode</h3>
        <ScopedToggleCard
          label="Privacy Mode"
          description="Master toggle: enables PII redaction and disables web search"
          settingsKey="privacyMode"
          checked={effectivePrivacy}
          onGlobalChange={async (v) => {
            setPrivacyMode(v);
            try {
              // Server profile is authoritative for privacy_mode.
              // Server applies privacy_mode side-effects (pii_redaction=true, web_search=false).
              await updateProfile({ privacy_mode: v });
            } catch (err) {
              console.error("Failed to sync privacy mode to server:", err);
            }
          }}
        />
      </div>

      {/* PII Redaction */}
      <div>
        <h3 className="text-sm font-medium text-white/80 mb-3">PII Redaction</h3>
        <ScopedToggleCard
          label="Redact Personal Information"
          description="Automatically redact emails, phone numbers, SSNs, and credit card numbers from transcripts"
          settingsKey="piiRedaction"
          checked={effectivePii}
          onGlobalChange={async (v) => {
            setPiiRedaction(v);
            try {
              await updateProfile({ pii_redaction: v });
            } catch (err) {
              console.error("Failed to sync PII redaction to server:", err);
            }
          }}
        />
        {effectivePrivacy && !effectivePii && (
          <p className="mt-1 text-xs text-amber-400">
            PII redaction is force-enabled by Privacy Mode.
          </p>
        )}
      </div>

      {/* Info */}
      <div className="flex items-start gap-2 rounded-lg border border-blue-500/20 bg-blue-500/5 px-4 py-3">
        <Shield className="h-4 w-4 text-blue-400 shrink-0 mt-0.5" />
        <p className="text-xs text-blue-200/70">
          When Privacy Mode is active, web search is disabled and all transcript text
          is automatically redacted for PII before processing. This helps protect
          sensitive information during conversations.
        </p>
      </div>
    </div>
  );
}

// ── Overlay Tab ──

function OverlayTab() {
  const { overlaySettings, setOverlaySettings } = useSettingsStore();

  const handleContentProtection = (enabled: boolean) => {
    setOverlaySettings({ contentProtection: enabled });
    window.electronAPI?.setContentProtection(enabled);
  };

  return (
    <div className="space-y-5">
      {/* Content protection */}
      <div>
        <h3 className="text-sm font-medium text-white/80 mb-3">Stream Protection</h3>
        <ToggleCard
          label="Hide from Screen Recordings"
          description="When enabled, the entire app is hidden from screen captures and streams"
          checked={overlaySettings.contentProtection}
          onChange={handleContentProtection}
        />
      </div>

      {/* Overlay mode */}
      <div>
        <h3 className="text-sm font-medium text-white/80 mb-3">Overlay Mode</h3>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setOverlaySettings({ overlayMode: "topbar" })}
            className={cn(
              "rounded-lg border px-4 py-3 text-left transition-colors",
              overlaySettings.overlayMode === "topbar"
                ? "border-blue-500/30 bg-blue-500/5 text-white/90"
                : "border-white/10 bg-white/[0.02] text-white/50 hover:bg-white/5",
            )}
          >
            <p className="text-sm font-medium mb-0.5">Top Bar</p>
            <p className="text-xs text-white/40">Slim bar at the top of your screen. Best for meetings.</p>
          </button>
          <button
            onClick={() => setOverlaySettings({ overlayMode: "cards" })}
            className={cn(
              "rounded-lg border px-4 py-3 text-left transition-colors",
              overlaySettings.overlayMode === "cards"
                ? "border-blue-500/30 bg-blue-500/5 text-white/90"
                : "border-white/10 bg-white/[0.02] text-white/50 hover:bg-white/5",
            )}
          >
            <p className="text-sm font-medium mb-0.5">Floating Cards</p>
            <p className="text-xs text-white/40">Draggable cards you can position anywhere. Best for interviews.</p>
          </button>
        </div>
      </div>

      {/* Section visibility */}
      <div>
        <h3 className="text-sm font-medium text-white/80 mb-3">Visible Sections</h3>
        <div className="space-y-2">
          <ToggleCard
            label="Transcript"
            description="Show recent transcript lines"
            checked={overlaySettings.showTranscript}
            onChange={(v) => setOverlaySettings({ showTranscript: v })}
          />
          <ToggleCard
            label="Fact Checks"
            description="Show fact-check badges"
            checked={overlaySettings.showFactChecks}
            onChange={(v) => setOverlaySettings({ showFactChecks: v })}
          />
          <ToggleCard
            label="Suggestions"
            description="Show AI response suggestions"
            checked={overlaySettings.showSuggestions}
            onChange={(v) => setOverlaySettings({ showSuggestions: v })}
          />
          <ToggleCard
            label="Action Items"
            description="Show extracted action items and notes"
            checked={overlaySettings.showNotes}
            onChange={(v) => setOverlaySettings({ showNotes: v })}
          />
        </div>
      </div>
    </div>
  );
}
