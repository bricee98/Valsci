(() => {
  const {
    escapeHtml,
    fetchJson,
    formatCurrency,
    hideStatus,
    setStatus,
    candidateStyle,
  } = window.ValsciUI;

  const config = window.arenaBuilderConfig || { providers: [], mockClaimSets: [] };
  const providerCatalog = [...(config.providers || [])];
  const stageNames = ["query_generation", "paper_analysis", "venue_scoring", "final_report"];
  const candidatePalette = ["#0f766e", "#c2410c", "#1d4ed8", "#b45309", "#be123c", "#4f46e5"];

  const byId = (id) => document.getElementById(id);

  let stagedClaims = [];
  let providerBlocks = [];
  let candidateEdits = {};
  let preflightPayload = null;
  let blockCounter = 0;

  function candidatePrefix(index) {
    return String.fromCharCode(65 + index);
  }

  function enabledModels(provider) {
    return (provider?.models || []).filter(model => model.enabled !== false);
  }

  function providerById(providerId) {
    return providerCatalog.find(provider => provider.provider_id === providerId) || providerCatalog[0] || null;
  }

  function invalidatePreflight() {
    preflightPayload = null;
    byId("launchArenaBtn").disabled = true;
  }

  function currentSearchConfig() {
    return {
      num_queries: parseInt(byId("numQueries").value || "5", 10),
      results_per_query: parseInt(byId("resultsPerQuery").value || "5", 10),
    };
  }

  function currentBibliometricConfig() {
    return {
      use_bibliometrics: byId("useBibliometrics").checked,
      author_impact_weight: 0.4,
      citation_impact_weight: 0.4,
      venue_impact_weight: 0.2,
    };
  }

  function parseBatchTags() {
    return byId("batchTags").value.split(",").map(value => value.trim()).filter(Boolean);
  }

  function providerBlockModelRows(block) {
    const provider = providerById(block.providerId);
    const search = String(block.search || "").trim().toLowerCase();
    return enabledModels(provider).filter(model => {
      const haystack = [
        model.label || model.model_name,
        model.model_name,
        model.context_window_tokens,
        model.input_cost_per_million,
        model.output_cost_per_million,
      ].join(" ").toLowerCase();
      return !search || haystack.includes(search);
    });
  }

  function candidateKey(blockId, modelName) {
    return `${blockId}::${modelName}`;
  }

  function derivedCandidates() {
    const candidates = [];
    providerBlocks.forEach(block => {
      const provider = providerById(block.providerId);
      if (!provider) {
        return;
      }
      enabledModels(provider)
        .filter(model => block.selectedModels.includes(model.model_name))
        .forEach(model => {
          const key = candidateKey(block.id, model.model_name);
          const edits = candidateEdits[key] || {
            label: model.label || model.model_name,
            overrides: stageNames.reduce((accumulator, stageName) => {
              accumulator[stageName] = model.model_name;
              return accumulator;
            }, {}),
          };
          candidateEdits[key] = edits;
          candidates.push({
            key,
            providerId: provider.provider_id,
            providerLabel: provider.label || provider.provider_id,
            provider,
            modelName: model.model_name,
            label: edits.label || model.label || model.model_name,
            overrides: { ...edits.overrides },
          });
        });
    });

    return candidates.map((candidate, index) => ({
      ...candidate,
      candidate_id: `candidate-${index}`,
      candidate_index: index,
      candidate_prefix: candidatePrefix(index),
      candidate_color: candidatePalette[index % candidatePalette.length],
    }));
  }

  function renderProviderBlocks() {
    const target = byId("providerBlocks");
    target.innerHTML = providerBlocks.map((block, index) => {
      const provider = providerById(block.providerId);
      const visibleModels = providerBlockModelRows(block);
      return `
        <section class="panel panel-muted" data-provider-block="${escapeHtml(block.id)}">
          <div class="panel-header">
            <div>
              <h3 class="panel-title">${providerBlocks.length > 1 ? `Provider ${index + 1}` : "Provider"}</h3>
              <p class="panel-subtitle">Choose a provider, then check the models to compare.</p>
            </div>
            <div class="inline-actions">
              ${providerBlocks.length > 1 ? `<button type="button" class="danger-button small-button" data-remove-provider="${escapeHtml(block.id)}">Remove</button>` : ""}
            </div>
          </div>

          <div class="grid-two">
            <div class="form-row">
              <label>Provider</label>
              <select data-provider-select="${escapeHtml(block.id)}">
                ${providerCatalog.map(item => `<option value="${escapeHtml(item.provider_id)}" ${item.provider_id === block.providerId ? "selected" : ""}>${escapeHtml(item.label || item.provider_id)}</option>`).join("")}
              </select>
            </div>
            <div class="form-row">
              <label>Find models</label>
              <input type="text" value="${escapeHtml(block.search || "")}" data-provider-search="${escapeHtml(block.id)}" placeholder="Filter models by name, size, or context">
            </div>
          </div>

          ${provider && !enabledModels(provider).length ? `
            <div class="empty-state">
              <strong>No models configured for this provider.</strong>
              <span>Add models on the <a href="/providers">Providers</a> page.</span>
            </div>
          ` : `
            <div class="comparison-scroll">
              <table class="model-table">
                <thead>
                  <tr>
                    <th>Use</th>
                    <th>Model</th>
                    <th>Context</th>
                    <th>Pricing</th>
                    <th>Capabilities</th>
                  </tr>
                </thead>
                <tbody>
                  ${visibleModels.map(model => `
                    <tr>
                      <td>
                        <label class="checkbox-row">
                          <input type="checkbox" data-model-toggle="${escapeHtml(block.id)}" value="${escapeHtml(model.model_name)}" ${block.selectedModels.includes(model.model_name) ? "checked" : ""}>
                          <span>Select</span>
                        </label>
                      </td>
                      <td>
                        <strong>${escapeHtml(model.label || model.model_name)}</strong>
                        <div class="helper-text">${escapeHtml(model.model_name)}</div>
                      </td>
                      <td>${Number(model.context_window_tokens || 0).toLocaleString()}</td>
                      <td>$${Number(model.input_cost_per_million || 0).toFixed(2)} in / $${Number(model.output_cost_per_million || 0).toFixed(2)} out per 1M</td>
                      <td>
                        <div class="pill-row">
                          ${model.supports_json_mode ? `<span class="pill">JSON</span>` : ""}
                          ${model.supports_temperature ? `<span class="pill">Temp</span>` : ""}
                          <span class="pill">${Number(model.max_output_tokens_default || 0).toLocaleString()} max out</span>
                        </div>
                      </td>
                    </tr>
                  `).join("") || `<tr><td colspan="5"><div class="empty-state"><strong>No models match this filter.</strong></div></td></tr>`}
                </tbody>
              </table>
            </div>
          `}
        </section>
      `;
    }).join("");
  }

  function renderClaims() {
    const target = byId("stagedClaims");
    if (!stagedClaims.length) {
      target.innerHTML = `<div class="empty-state"><strong>No claims staged yet.</strong><span>Add claims to the arena so preflight can estimate total runs and cost.</span></div>`;
      return;
    }
    target.innerHTML = stagedClaims.map((claim, index) => `
      <article class="staged-item">
        <div class="stack">
          <strong>Claim ${index + 1}</strong>
          <span>${escapeHtml(claim)}</span>
        </div>
        <button type="button" class="ghost-button small-button" data-remove-claim="${index}">Remove</button>
      </article>
    `).join("");
  }

  function renderCandidatePreview() {
    const candidates = derivedCandidates();
    const preview = byId("candidatePreview");
    const advanced = byId("advancedCandidateEditor");
    if (!candidates.length) {
      const emptyMarkup = `<div class="empty-state"><strong>No candidates yet.</strong><span>Select one or more models from a provider block to create comparison candidates automatically.</span></div>`;
      preview.innerHTML = emptyMarkup;
      advanced.innerHTML = emptyMarkup;
      renderLaunchReview();
      return;
    }

    preview.innerHTML = candidates.map(candidate => `
      <article class="record-card" style="${candidateStyle({ color: candidate.candidate_color })}">
        <div class="panel-header">
          <div class="candidate-chip">
            <span class="candidate-dot"></span>
            <strong>${escapeHtml(candidate.candidate_prefix)}</strong>
            <span>${escapeHtml(candidate.label)}</span>
          </div>
          <span class="badge neutral-badge">${escapeHtml(candidate.providerLabel)}</span>
        </div>
        <div class="record-meta">
          <span>${escapeHtml(candidate.modelName)}</span>
          <span>${escapeHtml(stageNames.every(stageName => candidate.overrides[stageName] === candidate.modelName) ? "Default mode" : "Advanced stage overrides")}</span>
        </div>
      </article>
    `).join("");

    advanced.innerHTML = candidates.map(candidate => {
      const providerModels = enabledModels(candidate.provider);
      return `
        <article class="record-card" data-candidate-editor="${escapeHtml(candidate.key)}" style="${candidateStyle({ color: candidate.candidate_color })}">
          <div class="panel-header">
            <div class="candidate-chip">
              <span class="candidate-dot"></span>
              <strong>${escapeHtml(candidate.candidate_prefix)}</strong>
              <span>${escapeHtml(candidate.providerLabel)}</span>
            </div>
            <span class="badge neutral-badge">${escapeHtml(candidate.modelName)}</span>
          </div>
          <div class="form-row">
            <label>Candidate Label</label>
            <input type="text" value="${escapeHtml(candidate.label)}" data-candidate-label="${escapeHtml(candidate.key)}">
          </div>
          <div class="grid-two">
            ${stageNames.map(stageName => `
              <div class="form-row">
                <label>${escapeHtml(window.ValsciUI.stageLabel(stageName))}</label>
                <select data-candidate-stage="${escapeHtml(candidate.key)}" data-stage-name="${escapeHtml(stageName)}">
                  ${providerModels.map(model => `<option value="${escapeHtml(model.model_name)}" ${candidate.overrides[stageName] === model.model_name ? "selected" : ""}>${escapeHtml(model.label || model.model_name)}</option>`).join("")}
                </select>
              </div>
            `).join("")}
          </div>
        </article>
      `;
    }).join("");

    renderLaunchReview();
  }

  function renderLaunchReview() {
    const candidates = derivedCandidates();
    const launchReview = byId("launchReview");
    if (!stagedClaims.length || !candidates.length) {
      launchReview.innerHTML = `<div class="empty-state"><strong>Select models and stage claims first.</strong><span>Preflight needs both claims and candidates before it can estimate the arena.</span></div>`;
      return;
    }
    if (!preflightPayload) {
      launchReview.innerHTML = `
        <div class="summary-strip">
          <div class="summary-cell"><span class="label">Unique Claims</span><span class="value">${new Set(stagedClaims).size}</span></div>
          <div class="summary-cell"><span class="label">Candidates</span><span class="value">${candidates.length}</span></div>
          <div class="summary-cell"><span class="label">Reuse Retrieval</span><span class="value">${byId("executionMode").value === "reuse_retrieval" ? "Enabled" : "Off"}</span></div>
        </div>
        <p class="helper-text">Estimate the arena to review total runs, expected cost, and duplicate handling before launch.</p>
      `;
      return;
    }

    const warnings = [];
    if (providerBlocks.length > 1) {
      warnings.push(`<div class="status-card info-card"><strong>Advanced multi-provider mode is active.</strong><span>This arena mixes models from multiple provider blocks, so candidate pricing and capabilities can vary more sharply across columns.</span></div>`);
    }
    if (preflightPayload.execution_mode === "reuse_retrieval" && candidates.length > 1) {
      warnings.push(`<div class="status-card warning-card"><strong>Candidate order matters in reuse mode.</strong><span>Candidate A supplies shared retrieval outputs for the later candidates in this round.</span></div>`);
    }
    if (!preflightPayload.totals.pricing_complete) {
      warnings.push(`<div class="status-card error-card"><strong>Pricing metadata is incomplete.</strong><span>${escapeHtml(preflightPayload.totals.missing_pricing_models.join(", "))}</span></div>`);
    }

    launchReview.innerHTML = `
      ${warnings.join("")}
      <div class="summary-strip">
        <div class="summary-cell"><span class="label">Unique Claims</span><span class="value">${preflightPayload.totals.unique_claim_count}</span></div>
        <div class="summary-cell"><span class="label">Runs</span><span class="value">${preflightPayload.totals.run_count}</span></div>
        <div class="summary-cell"><span class="label">Expected</span><span class="value">${formatCurrency(preflightPayload.totals.expected_cost_usd)}</span></div>
        <div class="summary-cell"><span class="label">Upper Bound</span><span class="value">${formatCurrency(preflightPayload.totals.upper_bound_cost_usd)}</span></div>
      </div>
      <div class="record-meta">
        <span>Execution mode: ${escapeHtml(preflightPayload.execution_mode)}</span>
        <span>Stop after: ${escapeHtml(preflightPayload.stop_after)}</span>
        <span>Duplicate handling: ${escapeHtml(byId("duplicateStrategy").value)}</span>
      </div>
    `;
    byId("launchArenaBtn").disabled = !preflightPayload.totals.pricing_complete;
  }

  function claimsFromTextarea() {
    return byId("claimInput").value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
  }

  function stageClaims(text) {
    const claims = String(text || "").split(/\r?\n/).map(value => value.trim()).filter(Boolean);
    claims.forEach(claim => {
      if (!stagedClaims.includes(claim)) {
        stagedClaims.push(claim);
      }
    });
    byId("claimInput").value = "";
    renderClaims();
    invalidatePreflight();
    renderLaunchReview();
  }

  function serializeCandidatesForApi() {
    return derivedCandidates().map(candidate => ({
      provider_id: candidate.providerId,
      label: candidate.label,
      candidate_id: candidate.candidate_id,
      candidate_index: candidate.candidate_index,
      candidate_prefix: candidate.candidate_prefix,
      candidate_color: candidate.candidate_color,
      model_overrides: { ...candidate.overrides },
    }));
  }

  async function estimateArena() {
    const candidates = serializeCandidatesForApi();
    if (!stagedClaims.length || !candidates.length) {
      throw new Error("Stage claims and select at least one model before estimating.");
    }
    const data = await fetchJson("/api/v1/claims/preflight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        claims: stagedClaims,
        candidates,
        search_config: currentSearchConfig(),
        execution_mode: byId("executionMode").value,
        stop_after: byId("stopAfter").value,
        duplicate_strategy: byId("duplicateStrategy").value,
      }),
    });
    preflightPayload = data;
    renderLaunchReview();
  }

  function openCostModal() {
    if (!preflightPayload) {
      throw new Error("Estimate the arena before launching it.");
    }
    const reuseNote = preflightPayload.execution_mode === "reuse_retrieval"
      ? `<div class="status-card warning-card"><strong>Reuse retrieval is enabled.</strong><span>Candidate A supplies shared query-generation and retrieval outputs for the later candidates in this round.</span></div>`
      : "";
    byId("costModalBody").innerHTML = `
      ${reuseNote}
      <div class="summary-strip">
        <div class="summary-cell"><span class="label">Unique Claims</span><span class="value">${preflightPayload.totals.unique_claim_count}</span></div>
        <div class="summary-cell"><span class="label">Runs</span><span class="value">${preflightPayload.totals.run_count}</span></div>
        <div class="summary-cell"><span class="label">Expected</span><span class="value">${formatCurrency(preflightPayload.totals.expected_cost_usd)}</span></div>
        <div class="summary-cell"><span class="label">Upper Bound</span><span class="value">${formatCurrency(preflightPayload.totals.upper_bound_cost_usd)}</span></div>
      </div>
    `;
    byId("confirmCostCheckbox").checked = false;
    byId("confirmCostBtn").disabled = true;
    byId("costModal").classList.remove("hidden");
  }

  async function launchArena() {
    const data = await fetchJson("/api/v1/arenas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: byId("arenaTitle").value.trim(),
        claims: stagedClaims,
        candidates: serializeCandidatesForApi(),
        batch_tags: parseBatchTags(),
        execution_mode: byId("executionMode").value,
        stop_after: byId("stopAfter").value,
        duplicate_strategy: byId("duplicateStrategy").value,
        search_config: currentSearchConfig(),
        bibliometric_config: currentBibliometricConfig(),
        cost_confirmation: {
          accepted: true,
          expected_cost_usd: preflightPayload.totals.expected_cost_usd,
          upper_bound_cost_usd: preflightPayload.totals.upper_bound_cost_usd,
        },
      }),
    });
    window.location.href = `/arena_results?arena_id=${encodeURIComponent(data.arena_id)}`;
  }

  async function refreshProviderModels(blockId) {
    const block = providerBlocks.find(item => item.id === blockId);
    if (!block) {
      return;
    }
    const provider = providerById(block.providerId);
    if (!provider) {
      return;
    }
    const data = await fetchJson("/api/v1/providers/ollama/discover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider_id: provider.provider_id }),
    });
    provider.models = mergeDiscoveredModels(provider.models || [], data.models || []);
    renderProviderBlocks();
    renderCandidatePreview();
    setStatus(byId("arenaStatus"), {
      title: "Models refreshed",
      message: `${provider.label || provider.provider_id} now has ${(provider.models || []).length} available model entries in this builder session.`,
      tone: "info",
    });
  }

  function mergeDiscoveredModels(existingModels, incomingModels) {
    const merged = new Map();
    existingModels.forEach(model => {
      merged.set(model.model_name, model);
    });
    incomingModels.forEach(model => {
      if (!merged.has(model.model_name)) {
        merged.set(model.model_name, model);
      }
    });
    return Array.from(merged.values());
  }

  function addProviderBlock() {
    providerBlocks.push({
      id: `provider-block-${blockCounter++}`,
      providerId: providerCatalog[0]?.provider_id || "",
      selectedModels: [],
      search: "",
    });
    renderProviderBlocks();
  }

  byId("addProviderBlockBtn").addEventListener("click", () => {
    addProviderBlock();
    invalidatePreflight();
  });
  byId("providerBlocks").addEventListener("change", event => {
    const providerSelect = event.target.closest("[data-provider-select]");
    if (providerSelect) {
      const block = providerBlocks.find(item => item.id === providerSelect.dataset.providerSelect);
      if (block) {
        block.providerId = providerSelect.value;
        block.selectedModels = [];
      }
      invalidatePreflight();
      renderProviderBlocks();
      renderCandidatePreview();
      return;
    }

    const modelToggle = event.target.closest("[data-model-toggle]");
    if (modelToggle) {
      const block = providerBlocks.find(item => item.id === modelToggle.dataset.modelToggle);
      if (!block) {
        return;
      }
      if (modelToggle.checked) {
        if (!block.selectedModels.includes(modelToggle.value)) {
          block.selectedModels.push(modelToggle.value);
        }
      } else {
        block.selectedModels = block.selectedModels.filter(value => value !== modelToggle.value);
      }
      invalidatePreflight();
      renderCandidatePreview();
      return;
    }

    const stageSelect = event.target.closest("[data-candidate-stage]");
    if (stageSelect) {
      const key = stageSelect.dataset.candidateStage;
      candidateEdits[key] = candidateEdits[key] || { label: "", overrides: {} };
      candidateEdits[key].overrides[stageSelect.dataset.stageName] = stageSelect.value;
      invalidatePreflight();
      renderCandidatePreview();
    }
  });
  byId("providerBlocks").addEventListener("input", event => {
    const searchInput = event.target.closest("[data-provider-search]");
    if (!searchInput) {
      return;
    }
    const block = providerBlocks.find(item => item.id === searchInput.dataset.providerSearch);
    if (!block) {
      return;
    }
    block.search = searchInput.value;
    renderProviderBlocks();
  });
  byId("providerBlocks").addEventListener("click", event => {
    const removeButton = event.target.closest("[data-remove-provider]");
    if (removeButton) {
      providerBlocks = providerBlocks.filter(item => item.id !== removeButton.dataset.removeProvider);
      invalidatePreflight();
      renderProviderBlocks();
      renderCandidatePreview();
    }
  });

  byId("advancedCandidateEditor").addEventListener("input", event => {
    const labelInput = event.target.closest("[data-candidate-label]");
    if (!labelInput) {
      return;
    }
    const key = labelInput.dataset.candidateLabel;
    candidateEdits[key] = candidateEdits[key] || { label: "", overrides: {} };
    candidateEdits[key].label = labelInput.value.trim();
    invalidatePreflight();
    renderCandidatePreview();
  });

  byId("stageClaimsBtn").addEventListener("click", () => stageClaims(byId("claimInput").value));
  byId("clearClaimsBtn").addEventListener("click", () => {
    stagedClaims = [];
    renderClaims();
    invalidatePreflight();
    renderLaunchReview();
  });
  byId("claimFile").addEventListener("change", async event => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    stageClaims(await file.text());
  });
  byId("stagedClaims").addEventListener("click", event => {
    const removeButton = event.target.closest("[data-remove-claim]");
    if (!removeButton) {
      return;
    }
    stagedClaims.splice(parseInt(removeButton.dataset.removeClaim, 10), 1);
    renderClaims();
    invalidatePreflight();
    renderLaunchReview();
  });

  document.querySelectorAll("[data-load-claim-set]").forEach(button => {
    button.addEventListener("click", () => {
      const claimSet = (config.mockClaimSets || []).find(item => item.pack_id === button.dataset.loadClaimSet);
      if (!claimSet) {
        return;
      }
      stagedClaims = [...claimSet.claims];
      renderClaims();
      invalidatePreflight();
      renderLaunchReview();
    });
  });

  [
    "arenaTitle",
    "batchTags",
    "executionMode",
    "stopAfter",
    "duplicateStrategy",
    "numQueries",
    "resultsPerQuery",
    "useBibliometrics",
  ].forEach(id => {
    byId(id).addEventListener("change", () => {
      invalidatePreflight();
      renderLaunchReview();
    });
    byId(id).addEventListener("input", () => {
      invalidatePreflight();
      renderLaunchReview();
    });
  });

  byId("estimateArenaBtn").addEventListener("click", () => {
    estimateArena().catch(error => {
      setStatus(byId("arenaStatus"), { title: "Arena estimate failed", message: error.message, tone: "error" });
    });
  });
  byId("launchArenaBtn").addEventListener("click", () => {
    try {
      openCostModal();
    } catch (error) {
      setStatus(byId("arenaStatus"), { title: "Launch blocked", message: error.message, tone: "error" });
    }
  });
  byId("cancelCostBtn").addEventListener("click", () => byId("costModal").classList.add("hidden"));
  byId("confirmCostCheckbox").addEventListener("change", event => {
    byId("confirmCostBtn").disabled = !event.target.checked;
  });
  byId("confirmCostBtn").addEventListener("click", () => {
    launchArena().catch(error => {
      setStatus(byId("arenaStatus"), { title: "Arena launch failed", message: error.message, tone: "error" });
    });
  });

  addProviderBlock();
  renderClaims();
  renderCandidatePreview();
  hideStatus(byId("arenaStatus"));
})();
