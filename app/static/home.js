(() => {
  const {
    escapeHtml,
    fetchJson,
    formatCurrency,
    formatDateTime,
    hideStatus,
    setStatus,
    revealPanel,
    flashButton,
    buttonBusy,
  } = window.ValsciUI;

  const config = window.homePageConfig || { providers: [] };
  const providerCatalog = config.providers || [];
  const stagedClaims = [];
  let preflightPayload = null;

  const stageFields = ["query_generation", "paper_analysis", "venue_scoring", "final_report"];

  const byId = (id) => document.getElementById(id);

  function currentProvider() {
    return providerCatalog.find(provider => provider.provider_id === byId("providerId").value) || providerCatalog[0] || null;
  }

  function enabledModels(provider) {
    return (provider?.models || []).filter(model => model.enabled !== false);
  }

  function syncProviderOptions() {
    const select = byId("providerId");
    if (!providerCatalog.length) {
      select.innerHTML = `<option value="">No providers enabled</option>`;
      select.disabled = true;
      byId("modelDefault").disabled = true;
      byId("stageClaimsBtn").disabled = true;
      byId("queueRunBtn").disabled = true;
      setStatus(byId("homeRunStatus"), {
        title: "No providers configured",
        message: "Open Providers from the top navigation, enable a provider, and add at least one model before running claims.",
        tone: "warning",
      });
      return;
    }
    hideStatus(byId("homeRunStatus"));
    select.disabled = false;
    byId("modelDefault").disabled = false;
    byId("stageClaimsBtn").disabled = false;
    byId("queueRunBtn").disabled = false;
    select.innerHTML = providerCatalog.map(provider => `
      <option value="${escapeHtml(provider.provider_id)}">${escapeHtml(provider.label || provider.provider_id)}</option>
    `).join("");
    syncModelOptions();
  }

  function syncModelOptions() {
    const models = enabledModels(currentProvider());
    if (!models.length) {
      byId("modelDefault").innerHTML = `<option value="">No models available</option>`;
      byId("modelDefault").disabled = true;
      return;
    }
    byId("modelDefault").disabled = false;
    byId("modelDefault").innerHTML = `
      <option value="" disabled selected>Select a model</option>
      ${models.map(model => `<option value="${escapeHtml(model.model_name)}">${escapeHtml(model.label || model.model_name)}</option>`).join("")}
    `;
  }

  function parseBatchTags() {
    return byId("batchTags").value
      .split(",")
      .map(value => value.trim())
      .filter(Boolean);
  }

  function buildCandidatesPayload() {
    const modelName = byId("modelDefault").value.trim();
    const modelOverrides = {};
    if (modelName) {
      stageFields.forEach(stage => {
        modelOverrides[stage] = modelName;
      });
    }
    return [
      {
        provider_id: byId("providerId").value,
        label: modelName || undefined,
        model_overrides: modelOverrides,
      },
    ];
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

  function invalidatePreflight() {
    preflightPayload = null;
    byId("preflightPanel").classList.add("hidden");
  }

  function loadClaimsIntoStage(text) {
    const claims = String(text || "")
      .split(/\r?\n/)
      .map(value => value.trim())
      .filter(Boolean);
    if (!claims.length) {
      setStatus(byId("homeRunStatus"), {
        title: "No claims found",
        message: "Paste one claim per line or upload a text file before staging.",
        tone: "warning",
      });
      return 0;
    }
    let addedCount = 0;
    let duplicateCount = 0;
    claims.forEach(claim => {
      if (!stagedClaims.includes(claim)) {
        stagedClaims.push(claim);
        addedCount += 1;
      } else {
        duplicateCount += 1;
      }
    });
    if (addedCount > 0) {
      byId("claimText").value = "";
    }
    renderStagedClaims();
    invalidatePreflight();
    setStatus(byId("homeRunStatus"), {
      title: addedCount ? "Claims staged" : "No new claims staged",
      message: `${addedCount} added${duplicateCount ? `, ${duplicateCount} duplicate${duplicateCount === 1 ? "" : "s"} skipped` : ""}.`,
      tone: addedCount ? "success" : "warning",
    });
    return addedCount;
  }

  function renderStagedClaims() {
    const target = byId("stagedClaims");
    if (!stagedClaims.length) {
      target.innerHTML = `<div class="empty-state"><strong>No claims staged yet.</strong><span>Paste claims or upload a text file to get started.</span></div>`;
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

  function renderPreflight(data) {
    const warnings = [];
    if (data.totals.duplicate_input_count) {
      warnings.push(`<div class="status-card warning-card"><strong>Duplicate inputs collapse before queueing.</strong><span>${data.totals.duplicate_input_count} repeated input(s) reduce to ${data.totals.unique_claim_count} unique claim(s).</span></div>`);
    }
    if (data.totals.reused_existing_count) {
      warnings.push(`<div class="status-card info-card"><strong>Existing history will be reused.</strong><span>${data.totals.reused_existing_count} claim(s) already exist and will open their existing run history instead of queueing a new run.</span></div>`);
    }
    if (!data.totals.pricing_complete) {
      warnings.push(`<div class="status-card error-card"><strong>Pricing metadata is incomplete.</strong><span>Missing pricing for ${escapeHtml(data.totals.missing_pricing_models.join(", "))}.</span></div>`);
    }
    byId("preflightPanel").innerHTML = `
      <div class="panel-header">
        <div>
          <h3 class="panel-title">Run Review</h3>
          <p class="panel-subtitle">One claim run per unique claim using the selected provider/model defaults.</p>
        </div>
      </div>
      ${warnings.join("")}
      <div class="summary-strip">
        <div class="summary-cell"><span class="label">Unique Claims</span><span class="value">${data.totals.unique_claim_count}</span></div>
        <div class="summary-cell"><span class="label">Runs</span><span class="value">${data.totals.run_count}</span></div>
        <div class="summary-cell"><span class="label">Expected</span><span class="value">${formatCurrency(data.totals.expected_cost_usd)}</span></div>
        <div class="summary-cell"><span class="label">Upper Bound</span><span class="value">${formatCurrency(data.totals.upper_bound_cost_usd)}</span></div>
      </div>
    `;
    revealPanel(byId("preflightPanel"));
  }

  async function runPreflight() {
    if (!stagedClaims.length) {
      throw new Error("Stage at least one claim before estimating.");
    }
    const data = await fetchJson("/api/v1/claims/preflight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        claims: stagedClaims,
        candidates: buildCandidatesPayload(),
        search_config: currentSearchConfig(),
        duplicate_strategy: byId("duplicateStrategy").value,
      }),
    });
    preflightPayload = data;
    renderPreflight(data);
    return data;
  }

  function openCostModal() {
    if (!preflightPayload) {
      throw new Error("Estimate the cost before queueing the run.");
    }
    byId("costModalBody").innerHTML = `
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

  async function queueRun() {
    const data = await fetchJson("/api/v1/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        claims: stagedClaims,
        candidates: buildCandidatesPayload(),
        search_config: currentSearchConfig(),
        bibliometric_config: currentBibliometricConfig(),
        batch_tags: parseBatchTags(),
        duplicate_strategy: byId("duplicateStrategy").value,
        cost_confirmation: {
          accepted: true,
          expected_cost_usd: preflightPayload.totals.expected_cost_usd,
          upper_bound_cost_usd: preflightPayload.totals.upper_bound_cost_usd,
        },
      }),
    });

    if (data.created_runs?.length) {
      const firstRun = data.created_runs[0];
      const claimParam = data.created_runs.length === 1 && firstRun.transport_claim_id
        ? `&claim_id=${encodeURIComponent(firstRun.transport_claim_id)}`
        : "";
      window.location.href = `/progress?batch_id=${encodeURIComponent(data.batch_id)}${claimParam}`;
      return;
    }

    if (data.reused_existing?.length === 1) {
      window.location.href = `/claims/${encodeURIComponent(data.reused_existing[0].claim_key)}?run_id=${encodeURIComponent(data.reused_existing[0].latest_run_id || "")}`;
      return;
    }

    window.location.href = "/browser";
  }

  function renderRecentArenas(arenas) {
    const target = byId("recentArenas");
    if (!arenas.length) {
      target.innerHTML = `<div class="empty-state"><strong>No arenas yet.</strong><span>Your recent comparisons will show up here once you launch them.</span></div>`;
      return;
    }
    target.innerHTML = arenas.map(arena => `
      <article class="arena-card">
        <div class="panel-header">
          <div>
            <h3 class="panel-title">${escapeHtml(arena.title)}</h3>
            <p class="panel-subtitle">${escapeHtml(arena.status.replace(/_/g, " "))} · ${escapeHtml(arena.current_stage_label || "")}</p>
          </div>
          <span class="badge neutral-badge">${arena.candidate_count} candidate${arena.candidate_count === 1 ? "" : "s"}</span>
        </div>
        <div class="arena-meta">
          <span>${arena.claim_count} claim${arena.claim_count === 1 ? "" : "s"}</span>
          <span>Updated ${escapeHtml(formatDateTime(arena.updated_at))}</span>
          <span>Expected ${escapeHtml(formatCurrency(arena.expected_cost_usd))}</span>
        </div>
        <div class="inline-actions">
          <a class="primary-button small-button" href="/arena_results?arena_id=${encodeURIComponent(arena.arena_id)}">Open Workspace</a>
        </div>
      </article>
    `).join("");
  }

  function renderRecentClaims(claims) {
    const target = byId("recentClaims");
    if (!claims.length) {
      target.innerHTML = `<div class="empty-state"><strong>No claim history yet.</strong><span>Queue a run and the latest claims will appear here.</span></div>`;
      return;
    }
    target.innerHTML = claims.map(claim => {
      const latestRun = claim.latest_run;
      const runSuffix = latestRun?.run_id ? `?run_id=${encodeURIComponent(latestRun.run_id)}` : "";
      return `
        <article class="record-card">
          <div class="stack">
            <strong>${escapeHtml(claim.text)}</strong>
            <div class="record-meta">
              <span>${claim.run_count} run${claim.run_count === 1 ? "" : "s"}</span>
              ${latestRun ? `<span>${escapeHtml(latestRun.rating_label || latestRun.status)}</span>` : "<span>No runs yet</span>"}
              ${latestRun?.last_activity_at ? `<span>${escapeHtml(formatDateTime(latestRun.last_activity_at))}</span>` : ""}
            </div>
          </div>
          <div class="inline-actions">
            <a href="/claims/${encodeURIComponent(claim.claim_key)}${runSuffix}" class="primary-button small-button">Open Claim</a>
          </div>
        </article>
      `;
    }).join("");
  }

  function dataBadge(label, tone) {
    const cls = {
      success: "success-badge",
      warning: "warning-badge",
      error: "error-badge",
      neutral: "neutral-badge",
    }[tone || "neutral"] || "neutral-badge";
    return `<span class="badge ${cls}">${escapeHtml(label)}</span>`;
  }

  function indexTone(indexStatus) {
    if (!indexStatus) return "neutral";
    if (indexStatus.state === "ready") return "success";
    if (indexStatus.state === "missing" || indexStatus.state === "needs_attention") return "warning";
    return "neutral";
  }

  function dataCoverageSummary(release) {
    if (!release) {
      return { label: "No release", tone: "warning", message: "Create a mini release or download a full release before running real-data checks." };
    }
    const coverage = release.manifest_coverage;
    if (coverage?.state === "stale") {
      const missing = Object.entries(coverage.missing || {})
        .map(([dataset, count]) => `${dataset}: ${Number(count || 0).toLocaleString()}`)
        .join(", ");
      return {
        label: "Rebuild needed",
        tone: "warning",
        message: missing ? `The current mini manifest has missing local rows (${missing}).` : "The selected mini release is stale versus the current manifest.",
      };
    }
    const missingDatasets = (release.datasets || []).filter(dataset => !dataset.exists || !dataset.file_count);
    if (missingDatasets.length) {
      return {
        label: "Incomplete",
        tone: "warning",
        message: `Missing dataset files: ${missingDatasets.map(dataset => dataset.name).join(", ")}.`,
      };
    }
    return { label: "Ready", tone: "success", message: "" };
  }

  function renderDataReadiness(state) {
    const target = byId("homeDataSummary");
    if (!target) return;
    const release = state.active_release || (state.releases || [])[0] || null;
    const coverage = dataCoverageSummary(release);
    const indexStatus = release?.index_status || null;
    const releaseLabel = release?.release_id || "None";
    const releaseType = release ? (release.is_mini ? "Mini" : "Full") : "";
    target.innerHTML = `
      <div class="summary-cell">
        <span class="label">Release</span>
        <span class="value data-path">${escapeHtml(releaseLabel)}</span>
        ${releaseType ? `<span class="record-meta">${escapeHtml(releaseType)}</span>` : ""}
      </div>
      <div class="summary-cell">
        <span class="label">Data</span>
        <span class="value">${dataBadge(coverage.label, coverage.tone)}</span>
      </div>
      <div class="summary-cell">
        <span class="label">Index</span>
        <span class="value">${dataBadge(indexStatus?.label || "Unknown", indexTone(indexStatus))}</span>
      </div>
    `;

    const statusTarget = byId("homeDataStatus");
    if (!statusTarget) return;
    if (coverage.message) {
      setStatus(statusTarget, {
        title: coverage.label,
        message: coverage.message,
        tone: coverage.tone,
      });
    } else if (!state.api_key_present) {
      setStatus(statusTarget, {
        title: "Semantic Scholar API key missing",
        message: "Existing data can be inspected, but live search and downloads need an API key.",
        tone: "warning",
      });
    } else {
      hideStatus(statusTarget);
    }
  }

  async function loadDataReadiness() {
    const data = await fetchJson("/api/v1/data/status");
    renderDataReadiness(data.state || {});
  }

  async function loadRecents() {
    const [arenaData, claimData] = await Promise.all([
      fetchJson("/api/v1/arenas?limit=4"),
      fetchJson("/api/v1/claims?limit=4"),
    ]);
    renderRecentArenas(arenaData.arenas || []);
    renderRecentClaims(claimData.claims || []);
  }

  async function migrateAll() {
    await fetchJson("/api/v1/migration/import_all", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ archive_after: true }),
    });
    window.location.href = "/migration";
  }

  byId("providerId")?.addEventListener("change", () => {
    syncModelOptions();
    invalidatePreflight();
  });
  byId("modelDefault")?.addEventListener("change", invalidatePreflight);
  byId("duplicateStrategy")?.addEventListener("change", invalidatePreflight);
  byId("batchTags")?.addEventListener("input", invalidatePreflight);
  byId("numQueries")?.addEventListener("input", invalidatePreflight);
  byId("resultsPerQuery")?.addEventListener("input", invalidatePreflight);
  byId("useBibliometrics")?.addEventListener("change", invalidatePreflight);

  byId("stageClaimsBtn")?.addEventListener("click", () => {
    const added = loadClaimsIntoStage(byId("claimText").value);
    flashButton(byId("stageClaimsBtn"), added
      ? { label: `Staged ${added} ✓` }
      : { label: "Nothing staged", tone: "error", duration: 1400 });
  });
  byId("clearClaimsBtn")?.addEventListener("click", () => {
    stagedClaims.splice(0, stagedClaims.length);
    renderStagedClaims();
    invalidatePreflight();
  });
  byId("claimFile")?.addEventListener("change", async event => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    loadClaimsIntoStage(await file.text());
  });
  byId("stagedClaims")?.addEventListener("click", event => {
    const button = event.target.closest("[data-remove-claim]");
    if (!button) {
      return;
    }
    stagedClaims.splice(parseInt(button.dataset.removeClaim, 10), 1);
    renderStagedClaims();
    invalidatePreflight();
  });
  byId("queueRunBtn")?.addEventListener("click", () => {
    const button = byId("queueRunBtn");
    // Step 1: estimate (skipped when a fresh estimate already exists),
    // Step 2: review and approve the cost in the confirmation modal.
    const openWhenPriced = () => {
      if (!preflightPayload.totals.pricing_complete) {
        setStatus(byId("homeRunStatus"), {
          title: "Pricing metadata is incomplete",
          message: `Missing pricing for ${preflightPayload.totals.missing_pricing_models.join(", ")}. Add it on the Providers page, then try again.`,
          tone: "error",
        });
        return;
      }
      hideStatus(byId("homeRunStatus"));
      openCostModal();
    };
    if (preflightPayload) {
      openWhenPriced();
      return;
    }
    const restore = buttonBusy(button, "Estimating…");
    runPreflight().then(() => {
      restore();
      openWhenPriced();
    }).catch(error => {
      restore();
      flashButton(button, { label: "Estimate failed ✗", tone: "error" });
      setStatus(byId("homeRunStatus"), { title: "Estimate failed", message: error.message, tone: "error" });
    });
  });
  byId("cancelCostBtn")?.addEventListener("click", () => byId("costModal").classList.add("hidden"));
  byId("confirmCostCheckbox")?.addEventListener("change", event => {
    byId("confirmCostBtn").disabled = !event.target.checked;
  });
  byId("confirmCostBtn")?.addEventListener("click", () => {
    const restore = buttonBusy(byId("confirmCostBtn"), "Queueing…");
    queueRun().catch(error => {
      restore();
      byId("costModal").classList.add("hidden");
      setStatus(byId("homeRunStatus"), { title: "Queue failed", message: error.message, tone: "error" });
    });
  });
  byId("migrateAllBtn")?.addEventListener("click", () => {
    const restore = buttonBusy(byId("migrateAllBtn"), "Migrating…");
    migrateAll().catch(error => {
      restore();
      flashButton(byId("migrateAllBtn"), { label: "Migration failed ✗", tone: "error" });
      setStatus(byId("homeRunStatus"), { title: "Migration failed", message: error.message, tone: "error" });
    });
  });

  syncProviderOptions();
  renderStagedClaims();
  loadDataReadiness().catch(error => {
    setStatus(byId("homeDataStatus"), { title: "Data status failed to load", message: error.message, tone: "error" });
  });
  loadRecents().catch(error => {
    setStatus(byId("homeRunStatus"), { title: "Home panels failed to load", message: error.message, tone: "error" });
  });
})();
