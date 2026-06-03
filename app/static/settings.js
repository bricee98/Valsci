(() => {
  const {
    escapeHtml,
    fetchJson,
    hideStatus,
    setStatus,
  } = window.ValsciUI;

  const byId = (id) => document.getElementById(id);
  const editorId = (key) => `env-editor-${key}`;
  const initialValues = new Map();
  let entries = [];

  function normalizeForCompare(value) {
    if (value && typeof value === "object") {
      return JSON.stringify(value);
    }
    return String(value ?? "");
  }

  function editorTextValue(entry) {
    const value = entry.value;
    if (entry.value_type === "object" || entry.value_type === "array") {
      const fallback = entry.value_type === "array" ? [] : {};
      return JSON.stringify(value ?? fallback, null, 2);
    }
    return value ?? "";
  }

  function renderEditor(entry) {
    const id = editorId(entry.env_key);
    if (entry.value_type === "boolean") {
      const checked = Boolean(entry.value);
      return `
        <label class="checkbox-row env-checkbox">
          <input type="checkbox" id="${escapeHtml(id)}" data-env-editor="${escapeHtml(entry.env_key)}" ${checked ? "checked" : ""}>
          <span>Enabled</span>
        </label>
      `;
    }
    if (entry.value_type === "object" || entry.value_type === "array") {
      return `
        <textarea
          id="${escapeHtml(id)}"
          data-env-editor="${escapeHtml(entry.env_key)}"
          spellcheck="false"
          class="env-json-input"
        >${escapeHtml(editorTextValue(entry))}</textarea>
      `;
    }
    if (entry.value_type === "integer" || entry.value_type === "number") {
      const step = entry.value_type === "integer" ? "1" : "any";
      return `
        <input
          type="number"
          id="${escapeHtml(id)}"
          data-env-editor="${escapeHtml(entry.env_key)}"
          step="${step}"
          value="${escapeHtml(editorTextValue(entry))}"
        >
      `;
    }
    const inputType = entry.sensitive ? "password" : "text";
    return `
      <div class="env-secret-row">
        <input
          type="${inputType}"
          id="${escapeHtml(id)}"
          data-env-editor="${escapeHtml(entry.env_key)}"
          value="${escapeHtml(editorTextValue(entry))}"
          autocomplete="off"
        >
        ${entry.sensitive ? `<button type="button" class="secondary-button small-button" data-reveal-env="${escapeHtml(entry.env_key)}">Reveal</button>` : ""}
      </div>
    `;
  }

  function renderEntries(state) {
    entries = state.entries || [];
    initialValues.clear();
    byId("envVarsPath").textContent = state.path || "Unknown";
    byId("envVarsExamplePath").textContent = state.example_path || "Unknown";
    byId("envVarsList").innerHTML = entries.map((entry) => {
      initialValues.set(entry.env_key, normalizeForCompare(entry.value));
      return `
        <article class="env-var-row" id="${escapeHtml(entry.env_key)}">
          <div class="env-var-key">
            <strong>${escapeHtml(entry.env_key)}</strong>
            <span>${entry.config_key ? `Config: ${escapeHtml(entry.config_key)}` : "Custom env var"}</span>
            <span class="badge ${entry.raw_present ? "success-badge" : "neutral-badge"}">${entry.raw_present ? "Configured" : "Default"}</span>
          </div>
          <div class="env-var-editor">
            ${renderEditor(entry)}
          </div>
          <div class="env-var-effective">
            <span class="label">Effective</span>
            <code>${escapeHtml(entry.effective_value)}</code>
            <span>${escapeHtml(entry.note || "")}</span>
          </div>
        </article>
      `;
    }).join("");
    updateDirtyState();
    focusHashTarget();
  }

  function parseEditorValue(entry) {
    const editor = byId(editorId(entry.env_key));
    if (!editor) {
      return entry.value;
    }
    if (entry.value_type === "boolean") {
      return editor.checked;
    }
    if (entry.value_type === "integer") {
      if (editor.value.trim() === "") {
        return "";
      }
      const value = Number.parseInt(editor.value, 10);
      if (!Number.isFinite(value)) {
        throw new Error(`${entry.env_key} must be an integer.`);
      }
      return value;
    }
    if (entry.value_type === "number") {
      if (editor.value.trim() === "") {
        return "";
      }
      const value = Number.parseFloat(editor.value);
      if (!Number.isFinite(value)) {
        throw new Error(`${entry.env_key} must be a number.`);
      }
      return value;
    }
    if (entry.value_type === "object" || entry.value_type === "array") {
      const parsed = JSON.parse(editor.value.trim() || (entry.value_type === "array" ? "[]" : "{}"));
      if (entry.value_type === "array" && !Array.isArray(parsed)) {
        throw new Error(`${entry.env_key} must be a JSON array.`);
      }
      if (entry.value_type === "object" && (!parsed || Array.isArray(parsed) || typeof parsed !== "object")) {
        throw new Error(`${entry.env_key} must be a JSON object.`);
      }
      return parsed;
    }
    return editor.value;
  }

  function currentSerializedValue(entry) {
    return normalizeForCompare(parseEditorValue(entry));
  }

  function updateDirtyState() {
    let dirty = false;
    for (const entry of entries) {
      try {
        dirty = dirty || currentSerializedValue(entry) !== initialValues.get(entry.env_key);
      } catch (_error) {
        dirty = true;
      }
    }
    byId("envVarsDirtyBadge").classList.toggle("hidden", !dirty);
    byId("saveEnvVarsBtn").disabled = !dirty;
  }

  function focusHashTarget() {
    const key = decodeURIComponent((window.location.hash || "").slice(1));
    if (!key) {
      return;
    }
    const row = byId(key);
    if (!row) {
      return;
    }
    row.classList.add("focused-env-var");
    row.scrollIntoView({ block: "center", behavior: "smooth" });
    const editor = byId(editorId(key));
    if (editor) {
      window.setTimeout(() => editor.focus(), 250);
    }
  }

  async function loadEnvVars() {
    const state = await fetchJson("/api/v1/settings/env");
    renderEntries(state);
    hideStatus(byId("envVarsStatus"));
  }

  async function saveEnvVars() {
    const updates = {};
    for (const entry of entries) {
      const value = parseEditorValue(entry);
      if (normalizeForCompare(value) !== initialValues.get(entry.env_key)) {
        updates[entry.env_key] = value;
      }
    }
    if (!Object.keys(updates).length) {
      setStatus(byId("envVarsStatus"), {
        title: "No changes to save",
        message: "The editor already matches env_vars.json.",
        tone: "info",
      });
      return;
    }
    const state = await fetchJson("/api/v1/settings/env", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ updates }),
    });
    renderEntries(state);
    setStatus(byId("envVarsStatus"), {
      title: "Configuration saved",
      message: "env_vars.json was updated. Direct config values are applied to the current web process; restart Valsci after changing paths or model routing defaults.",
      tone: "success",
    });
  }

  function bindEvents() {
    byId("reloadEnvVarsBtn").addEventListener("click", () => {
      loadEnvVars().catch((error) => setStatus(byId("envVarsStatus"), {
        title: "Reload failed",
        message: error.message,
        tone: "error",
      }));
    });
    byId("saveEnvVarsBtn").addEventListener("click", () => {
      saveEnvVars().catch((error) => setStatus(byId("envVarsStatus"), {
        title: "Save failed",
        message: error.message,
        tone: "error",
      }));
    });
    byId("envVarsList").addEventListener("input", updateDirtyState);
    byId("envVarsList").addEventListener("change", updateDirtyState);
    byId("envVarsList").addEventListener("click", (event) => {
      const button = event.target.closest("[data-reveal-env]");
      if (!button) {
        return;
      }
      const key = button.dataset.revealEnv;
      const editor = byId(editorId(key));
      if (!editor) {
        return;
      }
      editor.type = editor.type === "password" ? "text" : "password";
      button.textContent = editor.type === "password" ? "Reveal" : "Hide";
    });
    window.addEventListener("hashchange", focusHashTarget);
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    loadEnvVars().catch((error) => setStatus(byId("envVarsStatus"), {
      title: "Settings failed to load",
      message: error.message,
      tone: "error",
    }));
  });
})();
