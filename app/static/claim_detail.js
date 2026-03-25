(() => {
  const {
    escapeHtml,
    fetchJson,
    formatCurrency,
    formatDateTime,
    formatDurationMs,
    hideStatus,
    setStatus,
    stageLabel,
    candidateStyle,
    renderTransposedTable,
  } = window.ValsciUI;

  const config = window.claimDetailConfig || {};
  const providers = config.providers || [];
  const stageNames = ["query_generation", "paper_analysis", "venue_scoring", "final_report"];
  const byId = (id) => document.getElementById(id);

  let claimDetail = null;
  let comparisonSelection = new Set();
  let preflightPayload = null;

  function providerById(providerId) {
    return providers.find((provider) => provider.provider_id === providerId) || providers[0] || null;
  }

  function enabledModels(provider) {
    return (provider?.models || []).filter((model) => model.enabled !== false);
  }

  function formatTokens(value) {
    const amount = Number(value ?? 0);
    return Number.isFinite(amount) ? amount.toLocaleString() : "0";
  }

  function shortHash(value) {
    const text = String(value || "").trim();
    return text ? text.slice(0, 12) : "Pending";
  }

  function promptHashes(run) {
    const hashes = new Set();
    Object.values(run?.prompt_provenance || {}).forEach((entry) => {
      if (entry?.prompt_set_hash) {
        hashes.add(shortHash(entry.prompt_set_hash));
      }
    });
    return Array.from(hashes);
  }

  function promptComparison(run, focusedRun) {
    const left = promptHashes(focusedRun).join(",");
    const right = promptHashes(run).join(",");
    if (!left || !right) {
      return "Pending";
    }
    return left === right ? "Same prompt set" : "Prompt hash differs";
  }

  function stageOverrideSummary(run) {
    const overrides = run?.effective_models?.overrides || {};
    const pairs = Object.entries(overrides)
      .filter(([, model]) => model)
      .map(([stage, model]) => `${stageLabel(stage)}: ${model}`);
    return pairs.join(" | ") || "Provider default";
  }

  function modelSummary(run) {
    return Object.entries(run?.effective_models?.task_models || {})
      .map(([stage, model]) => `${stageLabel(stage)}: ${model}`)
      .join(" | ");
  }

  function searchConfigSummary(run) {
    const search = run?.search_config || {};
    return `${search.num_queries ?? 0} queries | ${search.results_per_query ?? 0} results/query`;
  }

  function bibliometricSummary(run) {
    const bibliometric = run?.bibliometric_config || {};
    if (!bibliometric.use_bibliometrics) {
      return "Disabled";
    }
    return `Enabled | author ${bibliometric.author_impact_weight ?? 0.4}, citation ${bibliometric.citation_impact_weight ?? 0.4}, venue ${bibliometric.venue_impact_weight ?? 0.2}`;
  }

  function reportHref(run) {
    if (!run) {
      return null;
    }
    if (run.is_stage_checkpoint && run.arena_id) {
      return `/arena_results?arena_id=${encodeURIComponent(run.arena_id)}&run_id=${encodeURIComponent(run.run_id)}`;
    }
    if (!run.transport_batch_id || !run.transport_claim_id) {
      return null;
    }
    if (run.report_available) {
      return `/results?batch_id=${encodeURIComponent(run.transport_batch_id)}&claim_id=${encodeURIComponent(run.transport_claim_id)}`;
    }
    return `/progress?batch_id=${encodeURIComponent(run.transport_batch_id)}&claim_id=${encodeURIComponent(run.transport_claim_id)}`;
  }

  function reportLabel(run) {
    if (!run) {
      return "Open";
    }
    if (run.is_stage_checkpoint && run.arena_id) {
      return "Open Arena Stage";
    }
    return run.report_available ? "Open Report" : "Open Progress";
  }

  function traceHref(run) {
    if (!run?.transport_batch_id || !run?.transport_claim_id) {
      return null;
    }
    // Hide trace link for runs that haven't started processing
    if (run.status === "queued") {
      return null;
    }
    return `/claims/${encodeURIComponent(run.transport_batch_id)}/${encodeURIComponent(run.transport_claim_id)}/trace`;
  }

  function claimSummary(run) {
    const report = run?.report || run?.claim_data?.report || {};
    return report.explanation || report.finalReasoning || "No final summary is available yet.";
  }

  function topEvidence(run) {
    const processed = run?.claim_data?.processed_papers || [];
    return processed.slice(0, 3).map((paper) => paper?.paper?.title || "Untitled paper");
  }

  function sourceContextMarkup() {
    const source = claimDetail?.source_context;
    if (!source || source.type !== "arena") {
      return "";
    }
    const arenaLabel = source.arena_title || source.arena_id;
    return `
      <div class="status-card info-card">
        <strong>Opened from Arena ${escapeHtml(arenaLabel)}</strong>
        <span>Focused on candidate ${escapeHtml(source.candidate_prefix || "?")} (${escapeHtml(source.candidate_label || "Unknown")}).</span>
      </div>
    `;
  }

  function normalizedPromptEntry(entry = {}) {
    const systemPrompt = entry.system_prompt || {};
    const userPrompt = entry.user_prompt || {};
    return {
      promptSetHash: entry.prompt_set_hash || null,
      renderedPromptHash: entry.rendered_prompt_hash || null,
      systemFile: entry.system_prompt_file || systemPrompt.file_name || null,
      userFile: entry.user_prompt_file || userPrompt.file_name || null,
      systemHash: entry.system_prompt_sha256 || systemPrompt.sha256 || null,
      userHash: entry.user_prompt_sha256 || userPrompt.sha256 || null,
      modifiedAt: entry.prompt_file_modified_at || systemPrompt.modified_at || userPrompt.modified_at || null,
    };
  }

  function renderPromptProvenance(run) {
    const provenance = run?.prompt_provenance || {};
    return `
      <details class="panel panel-muted prompt-provenance">
        <summary><strong>Prompt provenance</strong></summary>
        <div class="panel-divider"></div>
        <div class="record-list">
          ${stageNames.map((stage) => {
            const entry = normalizedPromptEntry(provenance[stage] || {});
            return `
              <article class="record-card">
                <div class="panel-header">
                  <div>
                    <h3 class="panel-title">${escapeHtml(stageLabel(stage))}</h3>
                    <p class="panel-subtitle">${escapeHtml(entry.systemFile || "No system prompt file recorded")}</p>
                  </div>
                  <span class="badge neutral-badge" title="Combined hash of system + user prompt templates for this stage">${escapeHtml(shortHash(entry.promptSetHash))}</span>
                </div>
                <div class="record-meta">
                  <span>User prompt: ${escapeHtml(entry.userFile || "Unknown")}</span>
                  <span title="Hash of the final prompt after variable substitution">Rendered hash: ${escapeHtml(shortHash(entry.renderedPromptHash))}</span>
                  <span>Modified: ${escapeHtml(formatDateTime(entry.modifiedAt))}</span>
                </div>
                <div class="pill-row">
                  <span class="pill" title="SHA256 of the system prompt template file">System hash ${escapeHtml(shortHash(entry.systemHash))}</span>
                  <span class="pill" title="SHA256 of the user prompt template file">User hash ${escapeHtml(shortHash(entry.userHash))}</span>
                </div>
              </article>
            `;
          }).join("")}
        </div>
      </details>
    `;
  }

  function renderFocusedRunHero() {
    const run = claimDetail?.focused_run;
    if (!run) {
      byId("focusedRunHero").innerHTML = `
        <div class="empty-state">
          <strong>Focused run not found.</strong>
          <span>This claim has no available runs to display.</span>
        </div>
      `;
      return;
    }

    document.title = `${claimDetail.claim.text} | Claim Detail`;

    const reportLink = reportHref(run);
    const traceLink = traceHref(run);
    const backToArena = claimDetail?.source_context?.arena_id
      ? `/arena_results?arena_id=${encodeURIComponent(claimDetail.source_context.arena_id)}&run_id=${encodeURIComponent(run.run_id)}`
      : null;
    const promptHashPills = promptHashes(run);

    byId("focusedRunHero").setAttribute("style", candidateStyle({ color: run.candidate_color }));
    byId("focusedRunHero").innerHTML = `
      <div class="stack">
        <div class="panel-header">
          <div class="stack">
            <div class="candidate-chip">
              <span class="candidate-dot"></span>
              <strong>${escapeHtml(run.candidate_prefix || "R")}</strong>
              <span>${escapeHtml(run.candidate_label || run.provider_label || run.run_id)}</span>
            </div>
            <h2 class="page-title claim-hero-title">${escapeHtml(claimDetail.claim.text)}</h2>
            <p class="page-subtitle">Claim key ${escapeHtml(claimDetail.claim.claim_key)} | ${escapeHtml(run.provider_label || run.provider_id || "Unknown provider")} | ${escapeHtml(run.default_model || "No default model")}</p>
          </div>
          <div class="header-actions">
            ${reportLink ? `<a class="primary-button small-button" href="${reportLink}">${escapeHtml(reportLabel(run))}</a>` : ""}
            ${traceLink ? `<a class="secondary-button small-button" href="${traceLink}">Open Trace</a>` : ""}
            <a class="ghost-button small-button" href="#comparisonMatrix">Compare With Alternatives</a>
            ${backToArena ? `<a class="ghost-button small-button" href="${backToArena}">Back to Arena</a>` : ""}
          </div>
        </div>
        ${sourceContextMarkup()}
        <div class="summary-strip">
          <div class="summary-cell">
            <span class="label">Status</span>
            <span class="value">${escapeHtml(run.status.replace(/_/g, " "))}</span>
          </div>
          <div class="summary-cell">
            <span class="label">Completed Stage</span>
            <span class="value">${escapeHtml(run.completed_stage_label || stageLabel(run.completed_stage || run.current_stage))}</span>
          </div>
          <div class="summary-cell">
            <span class="label">Rating</span>
            <span class="value">${escapeHtml(run.rating_label)}${run.claimRating !== null && run.claimRating !== undefined ? ` (${run.claimRating})` : ""}</span>
          </div>
          <div class="summary-cell">
            <span class="label">Actual Cost</span>
            <span class="value">${escapeHtml(formatCurrency(run.usage?.cost_usd || 0))}</span>
          </div>
        </div>
        <div class="detail-grid">
          <article class="panel panel-muted">
            <div class="panel-header">
              <div>
                <h3 class="panel-title">Runtime and Health</h3>
                <p class="panel-subtitle">Execution state, timing, token usage, and warning signals.</p>
              </div>
            </div>
            <div class="record-meta">
              <span>Current stage: ${escapeHtml(run.current_stage_label || stageLabel(run.current_stage))}</span>
              <span>Total elapsed: ${escapeHtml(formatDurationMs(run.total_elapsed_ms))}</span>
              <span>Last activity: ${escapeHtml(formatDateTime(run.last_activity_at || run.updated_at))}</span>
            </div>
            <div class="pill-row">
              <span class="pill">Issues ${run.quality_health?.issues_count || 0}</span>
              <span class="pill">Retries ${run.quality_health?.retry_count || 0}</span>
              <span class="pill">Truncation ${run.quality_health?.truncation_count || 0}</span>
              <span class="pill">Context warnings ${run.quality_health?.context_overflow_count || 0}</span>
              <span class="pill">Inaccessible papers ${run.quality_health?.inaccessible_papers_count || 0}</span>
              <span class="pill">Tokens ${escapeHtml(formatTokens(run.usage?.total_tokens || 0))}</span>
            </div>
          </article>
          <article class="panel panel-muted">
            <div class="panel-header">
              <div>
                <h3 class="panel-title">Models and Config</h3>
                <p class="panel-subtitle">Stage mappings, search configuration, bibliometric settings, and prompt hashes.</p>
              </div>
            </div>
            <div class="stack">
              <div class="helper-text"><strong>Stage models:</strong> ${escapeHtml(modelSummary(run) || "No model mapping recorded.")}</div>
              <div class="helper-text"><strong>Overrides:</strong> ${escapeHtml(stageOverrideSummary(run))}</div>
              <div class="helper-text"><strong>Search:</strong> ${escapeHtml(searchConfigSummary(run))}</div>
              <div class="helper-text"><strong>Bibliometrics:</strong> ${escapeHtml(bibliometricSummary(run))}</div>
              <div class="pill-row">
                ${promptHashPills.length ? promptHashPills.map((hash) => `<span class="pill">Prompt ${escapeHtml(hash)}</span>`).join("") : '<span class="pill">Prompt hashes pending</span>'}
              </div>
            </div>
          </article>
        </div>
        ${renderPromptProvenance(run)}
      </div>
    `;
  }

  function renderStageOutput() {
    const target = byId("stageOutput");
    const run = claimDetail?.focused_run;
    const cd = run?.claim_data || {};
    const queries = cd.semantic_scholar_queries || [];
    const papers = cd.processed_papers || [];
    const inaccessible = cd.inaccessible_papers || [];
    const report = cd.report || run?.report || {};
    const completedStage = run?.completed_stage || run?.current_stage;

    if (!run || (!queries.length && !papers.length && !report.explanation)) {
      target.classList.add("hidden");
      return;
    }
    target.classList.remove("hidden");

    const sections = [];

    // Queries section
    if (queries.length) {
      sections.push(`
        <details ${completedStage === "query_generation" ? "open" : ""}>
          <summary><strong>Generated Queries</strong> <span class="pill">${queries.length}</span></summary>
          <div class="panel-divider"></div>
          <ol style="margin:0;padding-left:20px">
            ${queries.map((q) => `<li style="margin-bottom:6px">${escapeHtml(q)}</li>`).join("")}
          </ol>
        </details>
      `);
    }

    // Papers section
    if (papers.length) {
      sections.push(`
        <details ${completedStage === "paper_analysis" || completedStage === "venue_scoring" ? "open" : ""}>
          <summary><strong>Reviewed Papers</strong> <span class="pill">${papers.length} relevant</span>${inaccessible.length ? ` <span class="pill">${inaccessible.length} inaccessible</span>` : ""}</summary>
          <div class="panel-divider"></div>
          <div class="record-list">
            ${papers.map((p) => {
              const paper = p.paper || p;
              const title = paper.title || "Untitled";
              const authors = (paper.authors || []).slice(0, 3).map((a) => a.name || a).join(", ");
              const year = paper.year || "";
              const score = p.score != null && p.score !== -1 ? p.score : null;
              const explanation = p.explanation || p.analysis_explanation || "";
              return `
                <article class="record-card">
                  <div class="stack">
                    <strong>${escapeHtml(title)}</strong>
                    <div class="record-meta">
                      ${authors ? `<span>${escapeHtml(authors)}</span>` : ""}
                      ${year ? `<span>${year}</span>` : ""}
                      ${score !== null ? `<span>Score: ${score}</span>` : ""}
                    </div>
                    ${explanation ? `<p class="helper-text">${escapeHtml(explanation)}</p>` : ""}
                  </div>
                </article>
              `;
            }).join("")}
          </div>
        </details>
      `);
    }

    // Report / Final output section
    if (report.explanation || report.finalReasoning) {
      const ratingText = run.rating_label ? `${run.rating_label}${run.claimRating != null ? ` (${run.claimRating})` : ""}` : "";
      sections.push(`
        <details ${completedStage === "final_report" ? "open" : ""}>
          <summary><strong>Final Report</strong>${ratingText ? ` <span class="pill">${escapeHtml(ratingText)}</span>` : ""}</summary>
          <div class="panel-divider"></div>
          <div class="stack">
            ${report.explanation ? `<div><strong>Summary</strong><p class="helper-text">${escapeHtml(report.explanation)}</p></div>` : ""}
            ${report.finalReasoning ? `<div><strong>Reasoning</strong><p class="helper-text">${escapeHtml(report.finalReasoning)}</p></div>` : ""}
          </div>
        </details>
      `);
    }

    target.innerHTML = `
      <div class="panel-header">
        <div>
          <h2 class="panel-title">Stage Output</h2>
          <p class="panel-subtitle">Results from each completed pipeline stage.</p>
        </div>
      </div>
      <div class="section-stack">
        ${sections.join("")}
      </div>
    `;
  }

  function defaultComparisonSelection() {
    comparisonSelection = new Set();
    (claimDetail?.alternative_runs || []).slice(0, 2).forEach((run) => {
      comparisonSelection.add(run.run_id);
    });
  }

  function renderAlternativeRuns() {
    const runs = claimDetail?.alternative_runs || [];
    const target = byId("alternativeRuns");
    if (!runs.length) {
      target.innerHTML = `
        <div class="empty-state">
          <strong>No alternative runs yet.</strong>
          <span>Queue a rerun from this page and it will appear here for side-by-side comparison.</span>
        </div>
      `;
      return;
    }

    target.innerHTML = runs.map((run) => {
      const reportLink = reportHref(run);
      const promptPill = promptComparison(run, claimDetail.focused_run);
      return `
        <article class="record-card" style="${candidateStyle({ color: run.candidate_color })}">
          <div class="panel-header">
            <div class="candidate-chip">
              <span class="candidate-dot"></span>
              <strong>${escapeHtml(run.candidate_prefix || "R")}</strong>
              <span>${escapeHtml(run.candidate_label || run.provider_label || run.run_id)}</span>
            </div>
            <span class="badge neutral-badge">${escapeHtml(run.rating_label)}</span>
          </div>
          <div class="record-meta">
            <span>${escapeHtml(run.provider_label || run.provider_id || "Unknown provider")}</span>
            <span>${escapeHtml(run.completed_stage_label || stageLabel(run.completed_stage || run.current_stage))}</span>
            <span>${escapeHtml(formatDateTime(run.updated_at))}</span>
          </div>
          <div class="pill-row">
            <span class="pill">${escapeHtml(promptPill)}</span>
            <span class="pill">Issues ${run.quality_health?.issues_count || 0}</span>
            <span class="pill">Actual ${escapeHtml(formatCurrency(run.usage?.cost_usd || 0))}</span>
          </div>
          <div class="inline-actions">
            <label class="checkbox-row">
              <input type="checkbox" data-compare-run="${escapeHtml(run.run_id)}" ${comparisonSelection.has(run.run_id) ? "checked" : ""}>
              <span>Compare</span>
            </label>
            <a href="/claims/${encodeURIComponent(claimDetail.claim.claim_key)}?run_id=${encodeURIComponent(run.run_id)}" class="secondary-button small-button">Focus Run</a>
            ${reportLink ? `<a href="${reportLink}" class="ghost-button small-button">${escapeHtml(reportLabel(run))}</a>` : ""}
          </div>
        </article>
      `;
    }).join("");
  }

  function comparisonRuns() {
    const focusedRun = claimDetail?.focused_run;
    const alternatives = (claimDetail?.alternative_runs || []).filter((run) => comparisonSelection.has(run.run_id));
    return [focusedRun, ...alternatives].filter(Boolean);
  }

  function renderComparisonMatrix() {
    const target = byId("comparisonMatrix");
    const runs = comparisonRuns();
    if (!claimDetail?.focused_run) {
      target.innerHTML = `<div class="empty-state"><strong>Nothing to compare yet.</strong><span>The focused run is missing.</span></div>`;
      return;
    }
    if (runs.length < 2) {
      target.innerHTML = `<div class="empty-state"><strong>Select one or more alternative runs.</strong><span>The focused run is pinned automatically. Add at least one alternative to open the matrix.</span></div>`;
      return;
    }

    const fastest = Math.min(...runs.map((r) => Number(r.total_elapsed_ms || 0)).filter((v) => v > 0));
    const cheapest = Math.min(...runs.map((r) => Number(r.usage?.cost_usd || 0)));
    const fewestIssues = Math.min(...runs.map((r) => Number(r.quality_health?.issues_count || 0)));

    const columns = [
      { label: "Provider", cell: (r) => escapeHtml(r.provider_label || r.provider_id || "Unknown") },
      { label: "Model", cell: (r) => escapeHtml(r.default_model || "Unknown") },
      { label: "Status", cell: (r) => escapeHtml(r.status.replace(/_/g, " ")) },
      { label: "Stage", cell: (r) => escapeHtml(stageLabel(r.current_stage)) },
      { label: "Rating", cell: (r) => escapeHtml(r.rating_label) + (r.claimRating != null ? ` (${r.claimRating})` : "") },
      { label: "Papers", cell: (r) => String((r.claim_data?.processed_papers || []).length) },
      { label: "Issues", cell: (r) => String(r.quality_health?.issues_count || 0), highlight: (r) => Number(r.quality_health?.issues_count || 0) === fewestIssues },
      { label: "Elapsed", cell: (r) => escapeHtml(formatDurationMs(r.total_elapsed_ms)), highlight: (r) => Number(r.total_elapsed_ms || 0) === fastest && fastest > 0 },
      { label: "Actual Cost", cell: (r) => escapeHtml(formatCurrency(r.usage?.cost_usd || 0)), highlight: (r) => Number(r.usage?.cost_usd || 0) === cheapest },
      { label: "Tokens", cell: (r) => escapeHtml(formatTokens(r.usage?.total_tokens || 0)) },
      { label: "Prompts", cell: (r) => escapeHtml(promptComparison(r, runs[0])) },
      { label: "", cell: (r) => `<a href="/claims/${encodeURIComponent(claimDetail.claim.claim_key)}?run_id=${encodeURIComponent(r.run_id)}" class="ghost-button small-button">Focus</a>` },
    ];

    const items = runs.map((r) => ({ ...r, _candidateColor: r.candidate_color }));

    target.innerHTML = renderTransposedTable({
      items,
      columns,
      rowHeader: (r) => `
        <div class="candidate-chip">
          <span class="candidate-dot"></span>
          <strong>${escapeHtml(r.candidate_prefix || "R")}</strong>
          <span>${escapeHtml(r.candidate_label || r.provider_label || r.run_id)}</span>
        </div>
      `,
      detailContent: (r) => {
        const summary = claimSummary(r);
        const evidence = topEvidence(r).join(" | ");
        if (!summary && !evidence) return null;
        return `
          ${summary ? `<div class="detail-card"><span class="label">Summary</span><span class="value">${escapeHtml(summary)}</span></div>` : ""}
          ${evidence ? `<div class="detail-card"><span class="label">Top Evidence</span><span class="value">${escapeHtml(evidence)}</span></div>` : ""}
        `;
      },
      focusTest: (r) => r.run_id === claimDetail.focused_run?.run_id,
    });

    // Wire detail row toggles
    target.querySelectorAll(".candidate-header").forEach((header) => {
      header.style.cursor = "pointer";
      header.addEventListener("click", () => {
        const detailRow = header.closest("tr").nextElementSibling;
        if (detailRow?.classList.contains("detail-row")) {
          detailRow.classList.toggle("open");
        }
      });
    });
  }

  function syncProviderOptions() {
    byId("providerId").innerHTML = providers.map((provider) => `
      <option value="${escapeHtml(provider.provider_id)}">${escapeHtml(provider.label || provider.provider_id)}</option>
    `).join("");
    syncModelOptions();
  }

  function syncModelOptions() {
    const models = enabledModels(providerById(byId("providerId").value));
    const currentValue = byId("modelDefault").value;
    byId("modelDefault").innerHTML = `
      <option value="">Provider Default</option>
      ${models.map((model) => `<option value="${escapeHtml(model.model_name)}">${escapeHtml(model.label || model.model_name)}</option>`).join("")}
    `;
    byId("modelDefault").value = models.some((model) => model.model_name === currentValue) ? currentValue : "";
  }

  function parseBatchTags() {
    return byId("batchTags").value
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);
  }

  function buildCandidatesPayload() {
    const providerId = byId("providerId").value;
    const modelName = byId("modelDefault").value.trim();
    const modelOverrides = {};
    if (modelName) {
      stageNames.forEach((stage) => {
        modelOverrides[stage] = modelName;
      });
    }
    return [
      {
        provider_id: providerId,
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
    byId("queueRunBtn").disabled = true;
    byId("preflightPanel").classList.add("hidden");
  }

  function renderPreflight(data) {
    const warnings = [];
    if (data.totals.duplicate_input_count) {
      warnings.push(`
        <div class="status-card warning-card">
          <strong>Duplicate inputs collapse before queueing.</strong>
          <span>${data.totals.duplicate_input_count} repeated input(s) reduce to ${data.totals.unique_claim_count} unique claim(s).</span>
        </div>
      `);
    }
    if (data.totals.reused_existing_count) {
      warnings.push(`
        <div class="status-card info-card">
          <strong>Existing history will be reused.</strong>
          <span>${data.totals.reused_existing_count} claim(s) already exist and will not queue a new run when duplicate handling is set to view.</span>
        </div>
      `);
    }
    if (!data.totals.pricing_complete) {
      warnings.push(`
        <div class="status-card error-card">
          <strong>Pricing metadata is incomplete.</strong>
          <span>Missing pricing for ${escapeHtml(data.totals.missing_pricing_models.join(", "))}.</span>
        </div>
      `);
    }
    byId("preflightPanel").classList.remove("hidden");
    byId("preflightPanel").innerHTML = `
      <div class="panel-header">
        <div>
          <h3 class="panel-title">Rerun Review</h3>
          <p class="panel-subtitle">One rerun will be created for the edited claim text unless an existing claim is reused.</p>
        </div>
      </div>
      ${warnings.join("")}
      <div class="summary-strip">
        <div class="summary-cell"><span class="label">Unique Claims</span><span class="value">${data.totals.unique_claim_count}</span></div>
        <div class="summary-cell"><span class="label">Runs</span><span class="value">${data.totals.run_count}</span></div>
        <div class="summary-cell"><span class="label">Expected</span><span class="value">${escapeHtml(formatCurrency(data.totals.expected_cost_usd))}</span></div>
        <div class="summary-cell"><span class="label">Upper Bound</span><span class="value">${escapeHtml(formatCurrency(data.totals.upper_bound_cost_usd))}</span></div>
      </div>
    `;
    byId("queueRunBtn").disabled = !data.totals.pricing_complete;
  }

  async function estimateRerun() {
    const claimText = byId("claimText").value.trim();
    if (!claimText) {
      throw new Error("Enter claim text before estimating a rerun.");
    }
    const data = await fetchJson("/api/v1/claims/preflight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        claims: [claimText],
        candidates: buildCandidatesPayload(),
        search_config: currentSearchConfig(),
        duplicate_strategy: byId("duplicateStrategy").value,
      }),
    });
    preflightPayload = data;
    renderPreflight(data);
  }

  function openCostModal() {
    if (!preflightPayload) {
      throw new Error("Estimate the rerun before queueing it.");
    }
    byId("costModalBody").innerHTML = `
      <div class="summary-strip">
        <div class="summary-cell"><span class="label">Unique Claims</span><span class="value">${preflightPayload.totals.unique_claim_count}</span></div>
        <div class="summary-cell"><span class="label">Runs</span><span class="value">${preflightPayload.totals.run_count}</span></div>
        <div class="summary-cell"><span class="label">Expected</span><span class="value">${escapeHtml(formatCurrency(preflightPayload.totals.expected_cost_usd))}</span></div>
        <div class="summary-cell"><span class="label">Upper Bound</span><span class="value">${escapeHtml(formatCurrency(preflightPayload.totals.upper_bound_cost_usd))}</span></div>
      </div>
    `;
    byId("confirmCostCheckbox").checked = false;
    byId("confirmCostBtn").disabled = true;
    byId("costModal").classList.remove("hidden");
  }

  async function queueRerun() {
    const data = await fetchJson("/api/v1/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        claims: [byId("claimText").value.trim()],
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
      const run = data.created_runs[0];
      window.location.href = `/claims/${encodeURIComponent(run.claim_key)}?run_id=${encodeURIComponent(run.run_id)}`;
      return;
    }
    if (data.reused_existing?.length === 1) {
      const existing = data.reused_existing[0];
      window.location.href = `/claims/${encodeURIComponent(existing.claim_key)}?run_id=${encodeURIComponent(existing.latest_run_id || "")}`;
      return;
    }
    window.location.href = `/claims/${encodeURIComponent(config.claimKey)}`;
  }

  function seedRerunForm() {
    const run = claimDetail?.focused_run;
    if (!run) {
      return;
    }
    byId("claimText").value = claimDetail.claim.text || "";
    byId("batchTags").value = (claimDetail.claim.batch_tags || []).join(", ");
    if (run.provider_id) {
      byId("providerId").value = run.provider_id;
      syncModelOptions();
    }
    const runModels = run.effective_models?.task_models || {};
    const allStageModels = new Set(Object.values(runModels).filter(Boolean));
    if (allStageModels.size === 1) {
      byId("modelDefault").value = Array.from(allStageModels)[0];
    }
    if (run.search_config?.num_queries) {
      byId("numQueries").value = run.search_config.num_queries;
    }
    if (run.search_config?.results_per_query) {
      byId("resultsPerQuery").value = run.search_config.results_per_query;
    }
    byId("useBibliometrics").checked = !!run.bibliometric_config?.use_bibliometrics;
  }

  async function loadClaimDetail() {
    const suffix = config.focusedRunId ? `?run_id=${encodeURIComponent(config.focusedRunId)}` : "";
    const data = await fetchJson(`/api/v1/claims/${encodeURIComponent(config.claimKey)}${suffix}`);
    claimDetail = data;
    defaultComparisonSelection();
    renderFocusedRunHero();
    renderStageOutput();
    renderAlternativeRuns();
    renderComparisonMatrix();
    seedRerunForm();
  }

  byId("alternativeRuns").addEventListener("change", (event) => {
    const checkbox = event.target.closest("[data-compare-run]");
    if (!checkbox) {
      return;
    }
    if (checkbox.checked) {
      comparisonSelection.add(checkbox.dataset.compareRun);
    } else {
      comparisonSelection.delete(checkbox.dataset.compareRun);
    }
    renderComparisonMatrix();
  });

  [
    "claimText",
    "providerId",
    "modelDefault",
    "duplicateStrategy",
    "batchTags",
    "numQueries",
    "resultsPerQuery",
    "useBibliometrics",
  ].forEach((id) => {
    byId(id).addEventListener("change", invalidatePreflight);
    byId(id).addEventListener("input", invalidatePreflight);
  });

  byId("providerId").addEventListener("change", () => {
    syncModelOptions();
    invalidatePreflight();
  });

  byId("estimateBtn").addEventListener("click", () => {
    estimateRerun().catch((error) => {
      setStatus(byId("rerunStatus"), { title: "Estimate failed", message: error.message, tone: "error" });
    });
  });

  byId("queueRunBtn").addEventListener("click", () => {
    try {
      openCostModal();
    } catch (error) {
      setStatus(byId("rerunStatus"), { title: "Queue blocked", message: error.message, tone: "error" });
    }
  });

  byId("cancelCostBtn").addEventListener("click", () => {
    byId("costModal").classList.add("hidden");
  });

  byId("confirmCostCheckbox").addEventListener("change", (event) => {
    byId("confirmCostBtn").disabled = !event.target.checked;
  });

  byId("confirmCostBtn").addEventListener("click", () => {
    queueRerun().catch((error) => {
      setStatus(byId("rerunStatus"), { title: "Queue failed", message: error.message, tone: "error" });
    });
  });

  syncProviderOptions();
  hideStatus(byId("claimDetailStatus"));
  hideStatus(byId("rerunStatus"));
  loadClaimDetail().catch((error) => {
    setStatus(byId("claimDetailStatus"), {
      title: "Claim detail failed to load",
      message: error.message,
      tone: "error",
    });
  });
})();
