(() => {
  const {
    escapeHtml,
    fetchJson,
    hideStatus,
    setStatus,
    flashButton,
    buttonBusy,
  } = window.ValsciUI;

  const byId = (id) => document.getElementById(id);
  const editorId = (key) => `env-editor-${key}`;
  const initialValues = new Map();
  let state = { groups: [], entries: [], routing_task_stages: [] };
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

  // ---- Editors -------------------------------------------------------------

  function renderBaseEditor(entry) {
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
      // Object/array values keep a raw JSON editor as the canonical source. For
      // LLM_ROUTING a friendly budget editor (rendered separately) drives it.
      const advanced = entry.env_key === "LLM_ROUTING";
      const textarea = `
        <textarea
          id="${escapeHtml(id)}"
          data-env-editor="${escapeHtml(entry.env_key)}"
          spellcheck="false"
          class="env-json-input"
        >${escapeHtml(editorTextValue(entry))}</textarea>
      `;
      if (advanced) {
        return `<details class="env-advanced-json"><summary>Advanced: edit raw JSON</summary>${textarea}</details>`;
      }
      return textarea;
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

  function renderRoutingBudgets() {
    const stages = state.routing_task_stages || [];
    const budgets = state.routing_output_budgets || {};
    const rows = stages.map((stage) => {
      const value = budgets[stage.key];
      return `
        <label class="budget-field">
          <span>${escapeHtml(stage.label)}</span>
          <input type="number" min="1" step="1" data-routing-budget="${escapeHtml(stage.key)}"
            value="${value == null ? "" : escapeHtml(String(value))}" placeholder="default">
        </label>
      `;
    }).join("");
    return `
      <div class="routing-budgets">
        <p class="budget-help"><strong>Advanced / optional.</strong> Per-stage override of a model's <strong>Max Output Tokens</strong>. Leave every field blank (the normal case) to use the value set per model on the <a href="/providers">Providers page</a> — that's the single place to set the output budget. Only fill these in if one stage needs a different budget than the others.</p>
        <div class="budget-grid">${rows}</div>
      </div>
    `;
  }

  function renderEntry(entry) {
    const id = editorId(entry.env_key);
    const isRouting = entry.env_key === "LLM_ROUTING";
    const restartBadge = entry.restart_required
      ? `<span class="badge warning-badge restart-badge" title="Read once at startup — restart the processor to apply">Restart</span>`
      : `<span class="badge success-badge auto-badge" title="The processor picks this up automatically within a few seconds of saving">Auto</span>`;
    const stateBadge = entry.raw_present
      ? `<span class="badge success-badge">Configured</span>`
      : `<span class="badge neutral-badge">Default</span>`;
    const desc = entry.description
      ? `<span class="env-desc">${escapeHtml(entry.description)}</span>`
      : "";
    const effective = entry.raw_present
      ? ""
      : `<span class="env-effective-inline">Effective: <code>${escapeHtml(String(entry.effective_value))}</code></span>`;
    return `
      <article class="setting-row" id="${escapeHtml(entry.env_key)}"
        data-setting-row="${escapeHtml(entry.env_key)}"
        data-search="${escapeHtml(((entry.label || "") + " " + entry.env_key + " " + (entry.description || "")).toLowerCase())}"
        data-configured="${entry.raw_present ? "1" : "0"}">
        <div class="setting-head">
          <label class="setting-label" for="${escapeHtml(id)}">${escapeHtml(entry.label || entry.env_key)}</label>
          <div class="setting-badges">${stateBadge}${restartBadge}</div>
        </div>
        <code class="setting-key">${escapeHtml(entry.env_key)}</code>
        ${desc}
        ${isRouting ? renderRoutingBudgets() : ""}
        <div class="setting-editor">${renderBaseEditor(entry)}</div>
        ${effective}
      </article>
    `;
  }

  function renderGroups() {
    entries = state.entries || [];
    initialValues.clear();
    byId("envVarsPath").textContent = state.path || "env_vars.json";
    entries.forEach((entry) => initialValues.set(entry.env_key, normalizeForCompare(entry.value)));

    const groups = state.groups || [];
    byId("settingsGroups").innerHTML = groups.map((group, index) => `
      <details class="panel settings-group" data-group="${escapeHtml(group.id)}" ${index < 3 ? "open" : ""}>
        <summary class="settings-group-summary">
          <div>
            <h2 class="panel-title">${escapeHtml(group.label)}</h2>
            <p class="panel-subtitle">${escapeHtml(group.description || "")}</p>
          </div>
          <span class="settings-group-count">${group.entries.length}</span>
        </summary>
        <div class="settings-group-body">
          ${group.entries.map(renderEntry).join("")}
        </div>
      </details>
    `).join("");

    syncRoutingBudgetsFromTextarea();
    updateDirtyState();
    applyFilter();
    focusHashTarget();
  }

  // ---- LLM_ROUTING budget editor <-> raw JSON sync -------------------------

  function routingTextarea() {
    return byId(editorId("LLM_ROUTING"));
  }

  function parseRoutingObject() {
    const textarea = routingTextarea();
    if (!textarea) return null;
    try {
      const parsed = JSON.parse(textarea.value.trim() || "{}");
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
    } catch (_error) {
      return null;
    }
  }

  // Push a budget number input into the raw LLM_ROUTING JSON textarea.
  function writeRoutingBudget(stageKey, rawValue) {
    const obj = parseRoutingObject();
    const budgetInputs = document.querySelectorAll("[data-routing-budget]");
    if (!obj) {
      // Malformed JSON: can't safely edit structurally — disable friendly inputs.
      budgetInputs.forEach((input) => { input.disabled = true; input.title = "Fix the raw JSON below to use these fields."; });
      return;
    }
    budgetInputs.forEach((input) => { input.disabled = false; input.title = ""; });
    if (!obj.tasks || typeof obj.tasks !== "object") obj.tasks = {};
    if (!obj.tasks[stageKey] || typeof obj.tasks[stageKey] !== "object") obj.tasks[stageKey] = {};
    const trimmed = String(rawValue).trim();
    if (trimmed === "") {
      delete obj.tasks[stageKey].max_output_tokens;
      if (Object.keys(obj.tasks[stageKey]).length === 0) delete obj.tasks[stageKey];
    } else {
      const num = Number.parseInt(trimmed, 10);
      if (Number.isFinite(num) && num > 0) {
        obj.tasks[stageKey].max_output_tokens = num;
      }
    }
    if (obj.tasks && Object.keys(obj.tasks).length === 0) delete obj.tasks;
    const textarea = routingTextarea();
    textarea.value = JSON.stringify(obj, null, 2);
    updateDirtyState();
  }

  // Refresh the friendly budget inputs from the current raw JSON (e.g. after a
  // manual textarea edit), so the two views never drift.
  function syncRoutingBudgetsFromTextarea() {
    const obj = parseRoutingObject();
    const budgetInputs = document.querySelectorAll("[data-routing-budget]");
    if (!obj) {
      budgetInputs.forEach((input) => { input.disabled = true; input.title = "Fix the raw JSON below to use these fields."; });
      return;
    }
    const tasks = obj.tasks && typeof obj.tasks === "object" ? obj.tasks : {};
    budgetInputs.forEach((input) => {
      input.disabled = false;
      input.title = "";
      const stageKey = input.dataset.routingBudget;
      const cfg = tasks[stageKey];
      const value = cfg && typeof cfg === "object" ? cfg.max_output_tokens : undefined;
      input.value = typeof value === "number" ? String(value) : "";
    });
  }

  // ---- Save / dirty tracking ----------------------------------------------

  function parseEditorValue(entry) {
    const editor = byId(editorId(entry.env_key));
    if (!editor) {
      return entry.value;
    }
    if (entry.value_type === "boolean") {
      return editor.checked;
    }
    if (entry.value_type === "integer") {
      if (editor.value.trim() === "") return "";
      const value = Number.parseInt(editor.value, 10);
      if (!Number.isFinite(value)) throw new Error(`${entry.label || entry.env_key} must be an integer.`);
      return value;
    }
    if (entry.value_type === "number") {
      if (editor.value.trim() === "") return "";
      const value = Number.parseFloat(editor.value);
      if (!Number.isFinite(value)) throw new Error(`${entry.label || entry.env_key} must be a number.`);
      return value;
    }
    if (entry.value_type === "object" || entry.value_type === "array") {
      let parsed;
      try {
        parsed = JSON.parse(editor.value.trim() || (entry.value_type === "array" ? "[]" : "{}"));
      } catch (_error) {
        throw new Error(`${entry.label || entry.env_key} is not valid JSON.`);
      }
      if (entry.value_type === "array" && !Array.isArray(parsed)) {
        throw new Error(`${entry.label || entry.env_key} must be a JSON array.`);
      }
      if (entry.value_type === "object" && (!parsed || Array.isArray(parsed) || typeof parsed !== "object")) {
        throw new Error(`${entry.label || entry.env_key} must be a JSON object.`);
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
    const restartKeys = entries
      .filter((entry) => entry.restart_required && entry.env_key in updates)
      .map((entry) => entry.label || entry.env_key);
    const newState = await fetchJson("/api/v1/settings/env", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ updates }),
    });
    state = newState;
    renderGroups();
    renderProcessorStatus(state.processor);
    const restartNote = restartKeys.length
      ? ` These need a processor restart to take effect: ${restartKeys.join(", ")}.`
      : " The processor will pick them up automatically within a few seconds.";
    setStatus(byId("envVarsStatus"), {
      title: "Settings saved",
      message: `Saved to env_vars.json and applied to the web app.${restartNote}`,
      tone: "success",
    });
  }

  // ---- Search / filter -----------------------------------------------------

  function applyFilter() {
    const query = (byId("settingsSearch").value || "").trim().toLowerCase();
    const changedOnly = byId("settingsChangedOnly").checked;
    let anyVisible = false;
    state.groups.forEach((group) => {
      const groupEl = document.querySelector(`[data-group="${group.id}"]`);
      if (!groupEl) return;
      let groupVisible = 0;
      group.entries.forEach((entry) => {
        const row = groupEl.querySelector(`[data-setting-row="${CSS.escape(entry.env_key)}"]`);
        if (!row) return;
        const matchesQuery = !query || row.dataset.search.includes(query);
        const matchesChanged = !changedOnly || row.dataset.configured === "1";
        const visible = matchesQuery && matchesChanged;
        row.style.display = visible ? "" : "none";
        if (visible) groupVisible += 1;
      });
      groupEl.style.display = groupVisible ? "" : "none";
      if (groupVisible && (query || changedOnly)) groupEl.open = true;
      anyVisible = anyVisible || groupVisible > 0;
    });
    byId("settingsEmptyState").style.display = anyVisible ? "none" : "";
  }

  function focusHashTarget() {
    const key = decodeURIComponent((window.location.hash || "").slice(1));
    if (!key) return;
    const row = byId(key);
    if (!row) return;
    const group = row.closest(".settings-group");
    if (group) group.open = true;
    row.classList.add("focused-env-var");
    row.scrollIntoView({ block: "center", behavior: "smooth" });
    const editor = byId(editorId(key));
    if (editor) window.setTimeout(() => editor.focus(), 250);
  }

  // ---- Live processor config-sync status ----------------------------------

  function renderProcessorStatus(status) {
    const card = byId("processorConfigStatus");
    if (!card || !status) return;
    let tone = "info-card";
    let title = "";
    let message = "";
    if (!status.alive) {
      tone = "error-card";
      title = "Processor is not running";
      message = "Saved settings will apply when the processor starts. Run \"python processor.py\", or \"docker compose restart processor\" if you deployed with Docker.";
    } else if (status.config_synced) {
      tone = "success-card";
      title = "Processor is up to date";
      message = "The background processor is running your latest saved settings.";
    } else {
      tone = "warning-card";
      title = "Applying your changes…";
      message = "The processor will pick up your saved settings within a few seconds (auto-reload). Settings marked Restart still need a processor restart.";
    }
    card.className = `status-card ${tone} settings-processor-status`;
    card.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(message)}</span>`;
  }

  async function pollProcessorStatus() {
    try {
      const status = await fetchJson("/api/v1/settings/processor-status");
      renderProcessorStatus(status);
    } catch (_error) {
      /* transient; keep last state */
    }
  }

  async function loadEnvVars() {
    state = await fetchJson("/api/v1/settings/env");
    renderGroups();
    renderProcessorStatus(state.processor);
    hideStatus(byId("envVarsStatus"));
  }

  function bindEvents() {
    byId("reloadEnvVarsBtn").addEventListener("click", () => {
      const button = byId("reloadEnvVarsBtn");
      const restore = buttonBusy(button, "Reloading…");
      loadEnvVars().then(() => {
        restore();
        flashButton(button, { label: "Reloaded ✓" });
      }).catch((error) => {
        restore();
        flashButton(button, { label: "Reload failed ✗", tone: "error" });
        setStatus(byId("envVarsStatus"), { title: "Reload failed", message: error.message, tone: "error" });
      });
    });

    byId("saveEnvVarsBtn").addEventListener("click", () => {
      const button = byId("saveEnvVarsBtn");
      const restore = buttonBusy(button, "Saving…");
      saveEnvVars().then(() => {
        restore();
        updateDirtyState();
        flashButton(button, { label: "Saved ✓" });
      }).catch((error) => {
        restore();
        flashButton(button, { label: "Save failed ✗", tone: "error" });
        setStatus(byId("envVarsStatus"), { title: "Save failed", message: error.message, tone: "error" });
      });
    });

    const groupsEl = byId("settingsGroups");
    groupsEl.addEventListener("input", (event) => {
      const budgetInput = event.target.closest("[data-routing-budget]");
      if (budgetInput) {
        writeRoutingBudget(budgetInput.dataset.routingBudget, budgetInput.value);
        return;
      }
      if (event.target.id === editorId("LLM_ROUTING")) {
        syncRoutingBudgetsFromTextarea();
      }
      updateDirtyState();
    });
    groupsEl.addEventListener("change", updateDirtyState);
    groupsEl.addEventListener("click", (event) => {
      const button = event.target.closest("[data-reveal-env]");
      if (!button) return;
      const editor = byId(editorId(button.dataset.revealEnv));
      if (!editor) return;
      editor.type = editor.type === "password" ? "text" : "password";
      button.textContent = editor.type === "password" ? "Reveal" : "Hide";
    });

    byId("settingsSearch").addEventListener("input", applyFilter);
    byId("settingsChangedOnly").addEventListener("change", applyFilter);
    window.addEventListener("hashchange", focusHashTarget);
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    loadEnvVars().catch((error) => setStatus(byId("envVarsStatus"), {
      title: "Settings failed to load",
      message: error.message,
      tone: "error",
    }));
    // Keep the processor sync status live so a save visibly transitions from
    // "Applying…" to "up to date" once the processor reloads.
    window.setInterval(pollProcessorStatus, 4000);
    window.addEventListener("beforeunload", () => {});
  });
})();
