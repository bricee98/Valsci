(() => {
  const { escapeHtml, fetchJson, formatDateTime, hideStatus, setStatus } = window.ValsciUI;
  const byId = (id) => document.getElementById(id);

  const PROVIDER_TYPES = [
    "openai", "openrouter", "ollama", "llamacpp", "local", "azure-openai", "azure-inference",
  ];

  const providerTypeMeta = {
    openai: {
      title: "OpenAI",
      hint: "OpenAI or compatible API.",
      connectionHint: "API key and optional base URL for OpenAI-compatible endpoints.",
      showApiKey: true,
      showBaseUrl: true,
      showLocalBackend: false,
      showOpenRouter: false,
      showAzureOpenAi: false,
      showAzureInference: false,
      defaultBaseUrl: "",
    },
    openrouter: {
      title: "OpenRouter",
      hint: "OpenRouter routing service.",
      connectionHint: "API key, base URL, and optional referer headers.",
      showApiKey: true,
      showBaseUrl: true,
      showLocalBackend: false,
      showOpenRouter: true,
      showAzureOpenAi: false,
      showAzureInference: false,
      defaultBaseUrl: "https://openrouter.ai/api/v1",
    },
    local: {
      title: "Local Router",
      hint: "Local compatibility layer.",
      connectionHint: "Base URL and optional backend type.",
      showApiKey: false,
      showBaseUrl: true,
      showLocalBackend: true,
      showOpenRouter: false,
      showAzureOpenAi: false,
      showAzureInference: false,
      defaultBaseUrl: "http://localhost:11434/v1",
    },
    ollama: {
      title: "Ollama",
      hint: "Direct Ollama instance.",
      connectionHint: "Base URL for the Ollama host (e.g. http://localhost:11434).",
      showApiKey: false,
      showBaseUrl: true,
      showLocalBackend: false,
      showOpenRouter: false,
      showAzureOpenAi: false,
      showAzureInference: false,
      defaultBaseUrl: "http://localhost:11434",
    },
    llamacpp: {
      title: "llama.cpp",
      hint: "Direct llama.cpp endpoint.",
      connectionHint: "Base URL for the llama.cpp server.",
      showApiKey: false,
      showBaseUrl: true,
      showLocalBackend: false,
      showOpenRouter: false,
      showAzureOpenAi: false,
      showAzureInference: false,
      defaultBaseUrl: "http://localhost:8080",
    },
    "azure-openai": {
      title: "Azure OpenAI",
      hint: "Azure OpenAI deployment.",
      connectionHint: "API key, resource endpoint, and API version.",
      showApiKey: true,
      showBaseUrl: false,
      showLocalBackend: false,
      showOpenRouter: false,
      showAzureOpenAi: true,
      showAzureInference: false,
      defaultBaseUrl: "",
    },
    "azure-inference": {
      title: "Azure AI Inference",
      hint: "Azure AI Inference endpoint.",
      connectionHint: "API key and inference endpoint.",
      showApiKey: true,
      showBaseUrl: false,
      showLocalBackend: false,
      showOpenRouter: false,
      showAzureOpenAi: false,
      showAzureInference: true,
      defaultBaseUrl: "",
    },
  };

  let providerCatalog = [];
  let selectedType = null;
  let selectedProviderId = null;
  let initialSnapshot = "";
  let discoveredOllamaModels = [];
  let discoverySourceLabel = "";

  function findEntryForType(type) {
    return providerCatalog.find((p) => p.provider_type === type);
  }

  function blankProviderForType(type) {
    const meta = providerTypeMeta[type] || providerTypeMeta.openai;
    return {
      provider_id: "",
      label: meta.title,
      provider_type: type,
      enabled: false,
      api_key: "",
      base_url: meta.defaultBaseUrl || "",
      default_model: "",
      local_backend: "",
      http_referer: "",
      site_name: "",
      azure_openai_endpoint: "",
      azure_openai_api_version: "2024-06-01",
      azure_ai_inference_endpoint: "",
      task_defaults: {},
      models: [],
    };
  }

  function normalizeModel(model = {}) {
    return {
      model_name: String(model.model_name || "").trim(),
      label: String(model.label || model.model_name || "").trim(),
      context_window_tokens: Number(model.context_window_tokens || 8192),
      max_output_tokens_default: Number(model.max_output_tokens_default || 1024),
      input_cost_per_million: Number(model.input_cost_per_million || 0),
      output_cost_per_million: Number(model.output_cost_per_million || 0),
      supports_temperature: model.supports_temperature !== false,
      supports_json_mode: model.supports_json_mode !== false,
      enabled: model.enabled !== false,
    };
  }

  function duplicateModelNames() {
    const seen = new Set();
    const duplicates = new Set();
    serializeModels().forEach((model) => {
      if (!model.model_name) return;
      if (seen.has(model.model_name)) duplicates.add(model.model_name);
      seen.add(model.model_name);
    });
    return duplicates;
  }

  // --- Sidebar rendering ---

  function renderTypeList() {
    const legacyEntries = [];
    const typeSeen = new Set();

    byId("providerList").innerHTML = PROVIDER_TYPES.map((type) => {
      const meta = providerTypeMeta[type];
      const entry = findEntryForType(type);
      const isEnabled = entry ? entry.enabled !== false : false;
      const isActive = type === selectedType;

      // Track duplicates
      const allEntries = providerCatalog.filter((p) => p.provider_type === type);
      if (allEntries.length > 1) {
        allEntries.slice(1).forEach((e) => legacyEntries.push(e));
      }

      return `
        <article class="record-card provider-type-row ${isActive ? "provider-type-row-active" : ""} ${!isEnabled ? "provider-type-row-disabled" : ""}" data-ptype="${escapeHtml(type)}">
          <div class="panel-header">
            <div>
              <strong>${escapeHtml(meta.title)}</strong>
              ${entry ? `<p class="panel-subtitle">${escapeHtml(entry.label || entry.provider_id)}</p>` : ""}
            </div>
            <label class="toggle-switch" title="${isEnabled ? "Disable" : "Enable"} ${escapeHtml(meta.title)}">
              <input type="checkbox" data-toggle-type="${escapeHtml(type)}" ${isEnabled ? "checked" : ""}>
              <span class="toggle-track"></span>
            </label>
          </div>
        </article>
      `;
    }).join("");

    if (legacyEntries.length) {
      byId("providerList").innerHTML += `
        <details class="panel panel-muted" style="margin-top:14px">
          <summary><strong>Legacy providers (${legacyEntries.length})</strong></summary>
          <div class="record-list" style="margin-top:10px">
            ${legacyEntries.map((entry) => `
              <article class="record-card" data-legacy-id="${escapeHtml(entry.provider_id)}">
                <div class="panel-header">
                  <div>
                    <strong>${escapeHtml(entry.label || entry.provider_id)}</strong>
                    <p class="panel-subtitle">${escapeHtml(entry.provider_id)} (${escapeHtml(entry.provider_type)})</p>
                  </div>
                  <button type="button" class="danger-button small-button" data-delete-legacy="${escapeHtml(entry.provider_id)}">Delete</button>
                </div>
              </article>
            `).join("")}
          </div>
        </details>
      `;
    }
  }

  // --- Editor ---

  function updateProviderContext() {
    const type = byId("providerType").value;
    const meta = providerTypeMeta[type] || providerTypeMeta.openai;
    byId("connectionHint").textContent = meta.connectionHint;

    byId("providerApiKeyGroup").classList.toggle("hidden", !meta.showApiKey);
    byId("providerBaseUrlGroup").classList.toggle("hidden", !meta.showBaseUrl);
    byId("providerLocalBackendGroup").classList.toggle("hidden", !meta.showLocalBackend);
    byId("openrouterSection").classList.toggle("hidden", !meta.showOpenRouter);
    byId("azureOpenAiSection").classList.toggle("hidden", !meta.showAzureOpenAi);
    byId("azureInferenceSection").classList.toggle("hidden", !meta.showAzureInference);

    // Ollama discovery only for ollama type
    const discoverySection = byId("ollamaDiscoverySection");
    if (discoverySection) {
      discoverySection.classList.toggle("hidden", type !== "ollama");
    }

    updateDiscoveryButtons();
  }

  function selectType(type) {
    selectedType = type;
    const entry = findEntryForType(type);
    if (entry) {
      populateEditor(entry);
    } else {
      populateEditor(blankProviderForType(type));
    }
    renderTypeList();
  }

  function populateEditor(provider) {
    const meta = providerTypeMeta[provider.provider_type] || providerTypeMeta.openai;
    selectedProviderId = provider.provider_id || null;
    selectedType = provider.provider_type;

    byId("editorTitle").textContent = meta.title;
    byId("editorSubtitle").textContent = selectedProviderId
      ? `Editing ${provider.label || selectedProviderId}.`
      : `New ${meta.title} provider. Save to create.`;

    byId("providerId").value = provider.provider_id || "";
    byId("providerType").value = provider.provider_type || "openai";
    byId("providerLabel").value = provider.label || "";
    byId("providerApiKey").value = provider.api_key || "";
    byId("providerBaseUrl").value = provider.base_url || "";
    byId("providerDefaultModel").value = provider.default_model || "";
    byId("providerLocalBackend").value = provider.local_backend || "";
    byId("providerHttpReferer").value = provider.http_referer || "";
    byId("providerSiteName").value = provider.site_name || "";
    byId("providerAzureOpenAiEndpoint").value = provider.azure_openai_endpoint || "";
    byId("providerAzureOpenAiApiVersion").value = provider.azure_openai_api_version || "2024-06-01";
    byId("providerAzureAiInferenceEndpoint").value = provider.azure_ai_inference_endpoint || "";
    const taskDefaults = provider.task_defaults || {};
    byId("taskDefaultQueryGeneration").value = taskDefaults.query_generation || "";
    byId("taskDefaultPaperAnalysis").value = taskDefaults.paper_analysis || "";
    byId("taskDefaultVenueScoring").value = taskDefaults.venue_scoring || "";
    byId("taskDefaultFinalReport").value = taskDefaults.final_report || "";
    byId("modelRows").innerHTML = "";
    (provider.models || []).forEach((model) => addModelRow(model));
    if (!(provider.models || []).length) addModelRow();

    // Show/hide delete button
    byId("deleteProviderBtn").classList.toggle("hidden", !selectedProviderId);

    updateProviderContext();
    initialSnapshot = snapshotEditor();
    updateDirtyState();
  }

  // --- Toggle handlers ---

  async function toggleProvider(type, enable) {
    const entry = findEntryForType(type);
    if (enable && !entry) {
      // Create new provider for this type
      const blank = blankProviderForType(type);
      blank.provider_id = type;
      blank.enabled = true;
      await fetchJson("/api/v1/providers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(blank),
      });
    } else if (entry) {
      // Update enabled state
      await fetchJson(`/api/v1/providers/${encodeURIComponent(entry.provider_id)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...entry, enabled: enable }),
      });
    }
    await loadProviders();
    if (enable) selectType(type);
  }

  // --- Model rows ---

  function addModelRow(model = {}) {
    const template = byId("modelRowTemplate");
    const fragment = template.content.cloneNode(true);
    const row = fragment.querySelector(".model-row");
    const normalized = normalizeModel(model);

    row.querySelector('[data-model-field="model_name"]').value = normalized.model_name;
    row.querySelector('[data-model-field="label"]').value = normalized.label;
    row.querySelector('[data-model-field="context_window_tokens"]').value = normalized.context_window_tokens;
    row.querySelector('[data-model-field="max_output_tokens_default"]').value = normalized.max_output_tokens_default;
    row.querySelector('[data-model-field="input_cost_per_million"]').value = normalized.input_cost_per_million;
    row.querySelector('[data-model-field="output_cost_per_million"]').value = normalized.output_cost_per_million;
    row.querySelector('[data-model-field="supports_temperature"]').checked = normalized.supports_temperature;
    row.querySelector('[data-model-field="supports_json_mode"]').checked = normalized.supports_json_mode;
    row.querySelector('[data-model-field="enabled"]').checked = normalized.enabled;

    byId("modelRows").appendChild(fragment);
    updateModelSummary();
  }

  function serializeModels() {
    return Array.from(document.querySelectorAll(".model-row")).map((row) => {
      const model = {
        model_name: row.querySelector('[data-model-field="model_name"]').value.trim(),
        label: row.querySelector('[data-model-field="label"]').value.trim(),
        context_window_tokens: Number(row.querySelector('[data-model-field="context_window_tokens"]').value || 8192),
        max_output_tokens_default: Number(row.querySelector('[data-model-field="max_output_tokens_default"]').value || 1024),
        input_cost_per_million: Number(row.querySelector('[data-model-field="input_cost_per_million"]').value || 0),
        output_cost_per_million: Number(row.querySelector('[data-model-field="output_cost_per_million"]').value || 0),
        supports_temperature: row.querySelector('[data-model-field="supports_temperature"]').checked,
        supports_json_mode: row.querySelector('[data-model-field="supports_json_mode"]').checked,
        enabled: row.querySelector('[data-model-field="enabled"]').checked,
      };
      return normalizeModel(model);
    }).filter((model) => model.model_name);
  }

  function updateModelSummary() {
    const models = serializeModels();
    const duplicateNames = duplicateModelNames();
    byId("modelCountBadge").textContent = `${models.length} model${models.length === 1 ? "" : "s"}`;
    document.querySelectorAll(".model-row").forEach((row) => {
      const modelName = row.querySelector('[data-model-field="model_name"]').value.trim();
      const label = row.querySelector('[data-model-field="label"]').value.trim() || modelName || "New model";
      const context = row.querySelector('[data-model-field="context_window_tokens"]').value || "0";
      row.querySelector("[data-model-title]").textContent = label;
      row.querySelector("[data-model-summary]").textContent = `${modelName || "Unnamed model"} | ${Number(context).toLocaleString()} context`;
      row.classList.toggle("warning-outline", duplicateNames.has(modelName));
    });
  }

  // --- Serialization ---

  function serializeProviderPayload() {
    return {
      provider_id: byId("providerId").value.trim() || selectedType || undefined,
      label: byId("providerLabel").value.trim() || undefined,
      provider_type: byId("providerType").value,
      api_key: byId("providerApiKey").value.trim() || undefined,
      base_url: byId("providerBaseUrl").value.trim() || undefined,
      default_model: byId("providerDefaultModel").value.trim() || undefined,
      local_backend: byId("providerLocalBackend").value.trim() || undefined,
      http_referer: byId("providerHttpReferer").value.trim() || undefined,
      site_name: byId("providerSiteName").value.trim() || undefined,
      azure_openai_endpoint: byId("providerAzureOpenAiEndpoint").value.trim() || undefined,
      azure_openai_api_version: byId("providerAzureOpenAiApiVersion").value.trim() || undefined,
      azure_ai_inference_endpoint: byId("providerAzureAiInferenceEndpoint").value.trim() || undefined,
      task_defaults: {
        query_generation: byId("taskDefaultQueryGeneration").value.trim() || undefined,
        paper_analysis: byId("taskDefaultPaperAnalysis").value.trim() || undefined,
        venue_scoring: byId("taskDefaultVenueScoring").value.trim() || undefined,
        final_report: byId("taskDefaultFinalReport").value.trim() || undefined,
      },
      models: serializeModels(),
    };
  }

  function snapshotEditor() {
    return JSON.stringify(serializeProviderPayload());
  }

  function updateDirtyState() {
    const dirty = snapshotEditor() !== initialSnapshot;
    byId("editorDirtyBadge").classList.toggle("hidden", !dirty);
  }

  // --- Discovery ---

  function visibleDiscoveredModels() {
    const filter = byId("ollamaSearch").value.trim().toLowerCase();
    return discoveredOllamaModels.filter((model) => {
      const meta = model.discovery_metadata || {};
      const haystack = [
        model.model_name, model.label, meta.tag, meta.size, meta.modified,
        meta.parameter_size, meta.family,
      ].join(" ").toLowerCase();
      return !filter || haystack.includes(filter);
    });
  }

  function renderDiscoveredModels(message = "") {
    const visible = visibleDiscoveredModels();
    const selectedCount = discoveredOllamaModels.filter((model) => model.selected).length;
    byId("ollamaImportCount").textContent = `${selectedCount} selected`;
    byId("ollamaImportStatus").textContent = message || `${visible.length} matching model(s), ${selectedCount} selected.`;
    byId("ollamaSourceLabel").textContent = discoverySourceLabel;
    byId("ollamaImportList").innerHTML = visible.map((model) => {
      const meta = model.discovery_metadata || {};
      return `
        <label class="record-card discovery-row">
          <div class="inline-actions">
            <input type="checkbox" data-discovered-model="${escapeHtml(model.model_name)}" ${model.selected ? "checked" : ""}>
            <div class="stack">
              <strong>${escapeHtml(model.model_name)}</strong>
              <div class="record-meta">
                ${meta.tag ? `<span>tag ${escapeHtml(meta.tag)}</span>` : ""}
                ${meta.size ? `<span>${escapeHtml(String(meta.size))}</span>` : ""}
                ${meta.modified ? `<span>${escapeHtml(formatDateTime(meta.modified))}</span>` : ""}
                <span>${Number(model.context_window_tokens || 0).toLocaleString()} context</span>
              </div>
              <div class="pill-row">
                ${meta.parameter_size ? `<span class="pill">${escapeHtml(meta.parameter_size)}</span>` : ""}
                ${meta.family ? `<span class="pill">${escapeHtml(meta.family)}</span>` : ""}
                ${(meta.capabilities || []).map((v) => `<span class="pill">${escapeHtml(v)}</span>`).join("")}
              </div>
            </div>
          </div>
        </label>
      `;
    }).join("") || `<div class="empty-state"><strong>No discovered models match the filter.</strong></div>`;
  }

  function updateDiscoveryButtons() {
    const providerHasUrl = !!byId("providerBaseUrl").value.trim();
    const arbitraryUrl = !!byId("discoveryBaseUrl").value.trim();
    const discoverBtn = byId("discoverProviderBtn");
    const probeBtn = byId("discoverUrlBtn");
    if (discoverBtn) discoverBtn.disabled = !providerHasUrl;
    if (probeBtn) probeBtn.disabled = !arbitraryUrl;
  }

  async function discoverModels(payload, sourceLabel) {
    const data = await fetchJson("/api/v1/providers/ollama/discover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    discoveredOllamaModels = (data.models || []).map((model) => ({
      ...normalizeModel(model),
      discovery_metadata: model.discovery_metadata || {},
      selected: true,
    }));
    discoverySourceLabel = sourceLabel;
    byId("ollamaSearch").value = "";
    renderDiscoveredModels();
    byId("ollamaImportModal").classList.remove("hidden");
  }

  function mergeDiscoveredModels({ replace = false } = {}) {
    const selectedModels = discoveredOllamaModels.filter((m) => m.selected).map((m) => normalizeModel(m));
    if (!selectedModels.length) {
      renderDiscoveredModels("No models selected.");
      return;
    }
    let merged = [];
    if (replace) {
      merged = selectedModels;
    } else {
      const existing = new Map();
      serializeModels().forEach((m) => existing.set(m.model_name, m));
      selectedModels.forEach((m) => { if (!existing.has(m.model_name)) existing.set(m.model_name, m); });
      merged = Array.from(existing.values());
    }
    byId("modelRows").innerHTML = "";
    merged.forEach((m) => addModelRow(m));
    if (!merged.length) addModelRow();
    byId("ollamaImportModal").classList.add("hidden");
    updateDirtyState();
    updateModelSummary();
    setStatus(byId("pageFeedback"), {
      title: replace ? "Replaced model rows" : "Merged model rows",
      message: `${selectedModels.length} model(s) imported. Save to persist.`,
      tone: "info",
    });
  }

  // --- Save / Delete / Load ---

  async function saveProvider() {
    const allRows = document.querySelectorAll(".model-row");
    const unnamed = Array.from(allRows).filter((row) => !row.querySelector('[data-model-field="model_name"]').value.trim());
    if (unnamed.length) {
      throw new Error(`${unnamed.length} model(s) have no name. Set a model name or remove them.`);
    }
    const duplicates = Array.from(duplicateModelNames());
    if (duplicates.length) {
      throw new Error(`Model names must be unique. Duplicates: ${duplicates.join(", ")}`);
    }
    const payload = serializeProviderPayload();
    // Preserve enabled state from existing entry
    const existingEntry = selectedProviderId ? providerCatalog.find((p) => p.provider_id === selectedProviderId) : null;
    if (existingEntry) {
      payload.enabled = existingEntry.enabled !== false;
    } else {
      payload.enabled = true;
    }
    const method = selectedProviderId ? "PUT" : "POST";
    const url = selectedProviderId
      ? `/api/v1/providers/${encodeURIComponent(selectedProviderId)}`
      : "/api/v1/providers";
    const saved = await fetchJson(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await loadProviders();
    selectType(saved.provider_type || selectedType);
    setStatus(byId("pageFeedback"), {
      title: "Provider saved",
      message: `${saved.label || saved.provider_id} saved.`,
      tone: "success",
    });
  }

  async function deleteProvider(providerId) {
    if (!window.confirm("Delete this provider? Its configuration will be lost.")) return;
    const entry = providerCatalog.find((p) => p.provider_id === providerId);
    await fetchJson(`/api/v1/providers/${encodeURIComponent(providerId)}`, { method: "DELETE" });
    await loadProviders();
    if (entry && entry.provider_type === selectedType) {
      selectType(selectedType);
    }
    setStatus(byId("pageFeedback"), {
      title: "Provider deleted",
      message: `${providerId} removed.`,
      tone: "warning",
    });
  }

  async function loadProviders() {
    const data = await fetchJson("/api/v1/providers");
    providerCatalog = data.providers || [];
    renderTypeList();
    // Re-populate editor if a type is selected
    if (selectedType) {
      const entry = findEntryForType(selectedType);
      if (entry) {
        populateEditor(entry);
      }
    }
  }

  // --- Event wiring ---

  // Sidebar: toggle to enable/disable
  byId("providerList").addEventListener("change", (event) => {
    const toggle = event.target.closest("[data-toggle-type]");
    if (!toggle) return;
    const type = toggle.dataset.toggleType;
    toggleProvider(type, toggle.checked).catch((error) => {
      toggle.checked = !toggle.checked;
      setStatus(byId("pageFeedback"), { title: "Toggle failed", message: error.message, tone: "error" });
    });
  });

  // Sidebar: click on type row to select
  byId("providerList").addEventListener("click", (event) => {
    // Don't handle clicks on toggle switches
    if (event.target.closest(".toggle-switch")) return;

    // Delete legacy entry
    const deleteLegacy = event.target.closest("[data-delete-legacy]");
    if (deleteLegacy) {
      deleteProvider(deleteLegacy.dataset.deleteLegacy).catch((error) => {
        setStatus(byId("pageFeedback"), { title: "Delete failed", message: error.message, tone: "error" });
      });
      return;
    }

    // Click on type row
    const row = event.target.closest("[data-ptype]");
    if (row) {
      selectType(row.dataset.ptype);
    }
  });

  // Editor input/change
  byId("providerEditor").addEventListener("input", (event) => {
    if (event.target.id === "providerBaseUrl" || event.target.id === "discoveryBaseUrl") {
      updateDiscoveryButtons();
    }
    if (event.target.closest(".model-row")) updateModelSummary();
    updateDirtyState();
  });

  byId("providerEditor").addEventListener("change", (event) => {
    if (event.target.closest(".model-row")) updateModelSummary();
    updateDirtyState();
  });

  byId("addModelBtn").addEventListener("click", () => {
    addModelRow();
    updateDirtyState();
  });

  byId("modelRows").addEventListener("click", (event) => {
    if (!event.target.closest("[data-remove-model]")) return;
    event.target.closest(".model-row").remove();
    if (!document.querySelector(".model-row")) addModelRow();
    else updateModelSummary();
    updateDirtyState();
  });

  // Discovery
  byId("discoverProviderBtn").addEventListener("click", () => {
    const btn = byId("discoverProviderBtn");
    const baseUrl = byId("providerBaseUrl").value.trim();
    if (!baseUrl) {
      setStatus(byId("pageFeedback"), { title: "No base URL", message: "Set a provider base URL first.", tone: "warning" });
      return;
    }
    const originalText = btn.textContent;
    btn.textContent = "Discovering...";
    btn.disabled = true;
    discoverModels(
      { base_url: baseUrl, api_key: byId("providerApiKey").value.trim() || undefined },
      `provider URL (${baseUrl})`
    ).catch((error) => {
      setStatus(byId("pageFeedback"), { title: "Discovery failed", message: error.message, tone: "error" });
    }).finally(() => {
      btn.textContent = originalText;
      btn.disabled = false;
    });
  });

  byId("discoverUrlBtn").addEventListener("click", () => {
    const btn = byId("discoverUrlBtn");
    const baseUrl = byId("discoveryBaseUrl").value.trim();
    if (!baseUrl) return;
    const originalText = btn.textContent;
    btn.textContent = "Probing...";
    btn.disabled = true;
    discoverModels(
      { base_url: baseUrl, api_key: byId("discoveryApiKey").value.trim() || undefined },
      `alternate URL (${baseUrl})`
    ).catch((error) => {
      setStatus(byId("pageFeedback"), { title: "Probe failed", message: error.message, tone: "error" });
    }).finally(() => {
      btn.textContent = originalText;
      btn.disabled = false;
    });
  });

  byId("ollamaSearch").addEventListener("input", () => renderDiscoveredModels());

  byId("ollamaImportList").addEventListener("change", (event) => {
    const checkbox = event.target.closest("[data-discovered-model]");
    if (!checkbox) return;
    const model = discoveredOllamaModels.find((m) => m.model_name === checkbox.dataset.discoveredModel);
    if (model) model.selected = checkbox.checked;
    renderDiscoveredModels();
  });

  byId("ollamaSelectVisibleBtn").addEventListener("click", () => {
    visibleDiscoveredModels().forEach((m) => { m.selected = true; });
    renderDiscoveredModels();
  });
  byId("ollamaClearVisibleBtn").addEventListener("click", () => {
    visibleDiscoveredModels().forEach((m) => { m.selected = false; });
    renderDiscoveredModels();
  });

  byId("ollamaAddModelsBtn").addEventListener("click", () => {
    const existingModels = serializeModels();
    const hasExisting = existingModels.some((m) => m.model_name);
    mergeDiscoveredModels({ replace: !hasExisting });
  });
  byId("ollamaImportCancelBtn").addEventListener("click", () => {
    byId("ollamaImportModal").classList.add("hidden");
  });

  // Save / Reset / Delete
  byId("saveProviderBtn").addEventListener("click", () => {
    saveProvider().catch((error) => {
      setStatus(byId("pageFeedback"), { title: "Save failed", message: error.message, tone: "error" });
      byId("pageFeedback").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  byId("resetProviderBtn").addEventListener("click", () => {
    if (selectedType) selectType(selectedType);
  });

  byId("deleteProviderBtn").addEventListener("click", () => {
    if (selectedProviderId) {
      deleteProvider(selectedProviderId).catch((error) => {
        setStatus(byId("pageFeedback"), { title: "Delete failed", message: error.message, tone: "error" });
      });
    }
  });

  // --- Init ---
  hideStatus(byId("pageFeedback"));
  loadProviders().then(() => {
    // Auto-select the first enabled provider, or the first type with an entry
    const firstEnabled = PROVIDER_TYPES.find((t) => {
      const entry = findEntryForType(t);
      return entry && entry.enabled !== false;
    });
    if (firstEnabled) selectType(firstEnabled);
  }).catch((error) => {
    setStatus(byId("pageFeedback"), { title: "Catalog load failed", message: error.message, tone: "error" });
  });
})();
