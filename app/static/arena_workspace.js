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

  const config = window.arenaWorkspaceConfig || {};
  const byId = (id) => document.getElementById(id);
  const orderedStages = ["query_generation", "paper_analysis", "venue_scoring", "final_report"];

  let arenaData = null;
  let arenaProgress = null;
  let continuationPreflight = null;
  let hasInitialized = false;
  let refreshTimer = null;
  let refreshInFlight = false;
  let lastRefreshCompletedAt = null;

  const autoRefreshMs = 5000;

  function nextStage(stage) {
    const index = orderedStages.indexOf(stage);
    if (index < 0 || index >= orderedStages.length - 1) {
      return null;
    }
    return orderedStages[index + 1];
  }

  function focusRun() {
    if (!config.focusRunId || !arenaData) {
      return null;
    }
    for (const group of arenaData.claim_groups || []) {
      const run = (group.runs || []).find((item) => item.run_id === config.focusRunId);
      if (run) {
        return run;
      }
    }
    return null;
  }

  function isFocusedRun(run) {
    return !!config.focusRunId && run?.run_id === config.focusRunId;
  }

  function statusTone(status) {
    const normalized = String(status || "").toLowerCase();
    if (["processed", "completed"].includes(normalized)) {
      return "success-badge";
    }
    if (normalized === "waiting_for_baseline") {
      return "warning-badge";
    }
    if (["error", "failed"].includes(normalized)) {
      return "error-badge";
    }
    return "neutral-badge";
  }

  function promptHashSummary(run, stage = arenaData?.current_stage) {
    const entry = run?.prompt_provenance?.[stage || "final_report"] || {};
    const hash = entry.prompt_set_hash || entry.rendered_prompt_hash;
    return hash ? hash.slice(0, 12) : "Pending";
  }

  function promptHashesForRun(run) {
    return Array.from(new Set(
      Object.values(run?.prompt_provenance || {})
        .map((entry) => entry?.prompt_set_hash || entry?.rendered_prompt_hash)
        .filter(Boolean)
        .map((hash) => String(hash).slice(0, 12))
    ));
  }

  function promptSummaryForRuns(runs) {
    const hashes = new Set();
    (runs || []).forEach((run) => {
      promptHashesForRun(run).forEach((hash) => hashes.add(hash));
    });
    if (!hashes.size) {
      return { label: "Prompt hashes pending", tone: "neutral-badge" };
    }
    if (hashes.size === 1) {
      return { label: "Same prompt set", tone: "success-badge" };
    }
    return { label: "Prompt changed", tone: "warning-badge" };
  }

  function claimReadyForDecision(group) {
    return (group.runs || []).every((run) => run.completed_stage === arenaData.current_stage);
  }

  function runSelectionName(group) {
    return `winner-${group.claim_key}`;
  }

  function sectionForClaimKey(claimKey) {
    return Array.from(document.querySelectorAll("[data-claim-group]")).find((section) => section.dataset.claimGroup === claimKey) || null;
  }

  function selectedRunIdForGroup(group) {
    const section = sectionForClaimKey(group.claim_key);
    if (!section) {
      return null;
    }
    const selected = section.querySelector(`input[name="${runSelectionName(group)}"]:checked`);
    return selected?.value || null;
  }

  function countClaimsWithEvidence(candidateId) {
    return (arenaData.claim_groups || []).reduce((count, group) => {
      const run = (group.runs || []).find((item) => item.candidate_id === candidateId);
      return count + (((run?.claim_data?.processed_papers || []).length > 0) ? 1 : 0);
    }, 0);
  }

  function renderSummaryStrip(summary) {
    byId("arenaSummaryStrip").innerHTML = `
      <div class="summary-cell">
        <span class="label">Status</span>
        <span class="value">${escapeHtml(summary.status.replace(/_/g, " "))}</span>
      </div>
      <div class="summary-cell">
        <span class="label">Current Stage</span>
        <span class="value">${escapeHtml(stageLabel(summary.current_stage))}</span>
      </div>
      <div class="summary-cell">
        <span class="label">Claims</span>
        <span class="value">${summary.claim_count}</span>
      </div>
      <div class="summary-cell">
        <span class="label">Candidates</span>
        <span class="value">${summary.candidate_count}</span>
      </div>
      <div class="summary-cell">
        <span class="label">Expected</span>
        <span class="value">${escapeHtml(formatCurrency(summary.expected_cost_usd))}</span>
      </div>
      <div class="summary-cell">
        <span class="label">Actual</span>
        <span class="value">${escapeHtml(formatCurrency(summary.actual_cost_usd))}</span>
      </div>
    `;
  }

  function stepState(key) {
    const currentStage = arenaData.current_stage;
    const currentIndex = orderedStages.indexOf(currentStage);
    const status = arenaData.summary?.status || "in_progress";

    if (key === "setup") {
      return "complete";
    }
    if (key === "running") {
      return status === "in_progress" ? "active" : "complete";
    }
    if (orderedStages.includes(key)) {
      const stepIndex = orderedStages.indexOf(key);
      if (stepIndex < currentIndex) {
        return "complete";
      }
      if (stepIndex === currentIndex) {
        return "active";
      }
      return "";
    }
    if (key === "select_winners") {
      // Only active after all pipeline stages are complete
      const lastStage = orderedStages[orderedStages.length - 1];
      const lastIndex = orderedStages.indexOf(lastStage);
      if (currentIndex < lastIndex) return "";
      const allReady = (arenaData.claim_groups || []).length > 0
        && (arenaData.claim_groups || []).every((group) => claimReadyForDecision(group));
      return allReady ? "active" : "";
    }
    if (key === "continue") {
      return continuationPreflight ? "active" : "";
    }
    return "";
  }

  function renderStepper() {
    const items = [
      { key: "setup", label: "Setup" },
      { key: "running", label: "Running" },
      { key: "query_generation", label: "Review Queries" },
      { key: "paper_analysis", label: "Review Evidence" },
      { key: "venue_scoring", label: "Review Scores" },
      { key: "final_report", label: "Review Final Reports" },
      { key: "select_winners", label: "Select Winners" },
      { key: "continue", label: "Continue" },
    ];
    byId("arenaStepper").innerHTML = items.map((item) => {
      const state = stepState(item.key);
      return `
        <div class="step-item ${state}">
          <strong>${escapeHtml(item.label)}</strong>
          <span>${state === "active" ? "Current focus" : state === "complete" ? "Complete" : "Upcoming"}</span>
        </div>
      `;
    }).join("");
  }

  function renderOverview() {
    const candidates = arenaProgress?.candidates || [];
    const overview = byId("overviewContent");
    if (!candidates.length) {
      overview.innerHTML = `<div class="empty-state"><strong>No candidate progress yet.</strong><span>Launch runs from the arena builder.</span></div>`;
      return;
    }

    const focused = focusRun();
    const fastest = Math.min(...candidates.map((c) => Number(c.total_elapsed_ms || 0)).filter((v) => v > 0));
    const cheapest = Math.min(...candidates.map((c) => Number(c.actual_cost_usd || 0)));
    const fewestWarnings = Math.min(...candidates.map((c) => Number(c.issue_count || 0)));
    const evidenceByCandidate = candidates.reduce((acc, c) => {
      acc[c.candidate_id] = countClaimsWithEvidence(c.candidate_id);
      return acc;
    }, {});
    const mostEvidence = Math.max(...Object.values(evidenceByCandidate), 0);

    const columns = [
      { label: "Provider", cell: (c) => escapeHtml(c.provider_label || c.provider_id || "Unknown") },
      { label: "Model", cell: (c) => escapeHtml(c.default_model || "Unknown") },
      { label: "Status", cell: (c) => `<span class="badge ${statusTone(c.status)}">${escapeHtml(c.status.replace(/_/g, " "))}</span>` },
      { label: "Stage", cell: (c) => escapeHtml(stageLabel(c.current_stage)) },
      { label: "Elapsed", cell: (c) => escapeHtml(formatDurationMs(c.total_elapsed_ms)), highlight: (c) => Number(c.total_elapsed_ms || 0) === fastest && fastest > 0 },
      { label: "Query", cell: (c) => escapeHtml(formatDurationMs(c.stage_timings_ms?.query_generation)) },
      { label: "Evidence", cell: (c) => escapeHtml(formatDurationMs(c.stage_timings_ms?.paper_analysis)) },
      { label: "Scoring", cell: (c) => escapeHtml(formatDurationMs(c.stage_timings_ms?.venue_scoring)) },
      { label: "Report", cell: (c) => escapeHtml(formatDurationMs(c.stage_timings_ms?.final_report)) },
      { label: "Issues", cell: (c) => String(c.issue_count || 0), highlight: (c) => Number(c.issue_count || 0) === fewestWarnings },
      { label: "Evidence", cell: (c) => String(evidenceByCandidate[c.candidate_id] || 0), highlight: (c) => evidenceByCandidate[c.candidate_id] === mostEvidence && mostEvidence > 0 },
      { label: "Actual Cost", cell: (c) => escapeHtml(formatCurrency(c.actual_cost_usd)), highlight: (c) => Number(c.actual_cost_usd || 0) === cheapest },
    ];

    const items = candidates.map((c) => ({ ...c, _candidateColor: c.candidate?.color }));

    overview.innerHTML = `
      ${focused ? `
        <div class="status-card info-card">
          <strong>Focused run highlighted</strong>
          <span>${escapeHtml(focused.candidate_prefix || "R")} for claim "${escapeHtml(focused.text)}" is highlighted.</span>
        </div>
      ` : ""}
      ${renderTransposedTable({
        items,
        columns,
        rowHeader: (c) => `
          <div class="candidate-chip">
            <span class="candidate-dot"></span>
            <strong>${escapeHtml(c.candidate?.prefix || "?")}</strong>
            <span>${escapeHtml(c.candidate?.label || c.candidate_id)}</span>
          </div>
        `,
        focusTest: (c) => focused && focused.candidate_id === c.candidate_id,
      })}
    `;
  }

  function stageRows(run) {
    const claimData = run.claim_data || {};
    if (arenaData.current_stage === "query_generation") {
      const queries = claimData.semantic_scholar_queries || [];
      return [
        { label: "Generated queries", value: String(queries.length) },
        { label: "Distinct queries", value: String(new Set(queries).size) },
        { label: "Preview", value: queries.slice(0, 3).join(" | ") || "No queries generated" },
      ];
    }
    if (arenaData.current_stage === "paper_analysis") {
      const processed = claimData.processed_papers || [];
      const inaccessible = claimData.inaccessible_papers || [];
      return [
        { label: "Relevant papers", value: String(processed.length) },
        { label: "Inaccessible", value: String(inaccessible.length) },
        {
          label: "Top evidence",
          value: processed.slice(0, 3).map((paper) => paper?.paper?.title || "Untitled").join(" | ") || "No evidence yet",
        },
      ];
    }
    if (arenaData.current_stage === "venue_scoring") {
      const processed = claimData.processed_papers || [];
      return [
        {
          label: "Scored papers",
          value: String(processed.filter((paper) => paper?.score !== undefined && paper?.score !== null && paper?.score !== -1).length),
        },
        { label: "Bibliometrics", value: run.bibliometric_config?.use_bibliometrics ? "Enabled" : "Off" },
        {
          label: "Ordering preview",
          value: processed.slice(0, 3).map((paper) => `${paper?.paper?.title || "Untitled"} (${paper?.score ?? "n/a"})`).join(" | ") || "No scoring yet",
        },
      ];
    }
    const report = run.report || claimData.report || {};
    return [
      {
        label: "Rating",
        value: `${run.rating_label}${run.claimRating !== null && run.claimRating !== undefined ? ` (${run.claimRating})` : ""}`,
      },
      { label: "Summary", value: report.explanation || "No explanation yet" },
      { label: "Final reasoning", value: report.finalReasoning || "Pending" },
    ];
  }

  function renderClaims() {
    const target = byId("claimGroups");
    const groups = arenaData.claim_groups || [];
    if (!groups.length) {
      target.innerHTML = `<div class="empty-state"><strong>No claims found in this arena round.</strong><span>Return to the builder to stage claims.</span></div>`;
      return;
    }

    target.innerHTML = groups.map((group) => {
      const ready = claimReadyForDecision(group);
      const focused = (group.runs || []).some((run) => isFocusedRun(run));
      const defaultSelection = ready && group.runs.length === 1 ? group.runs[0].run_id : "";

      // Build short metric columns from stageRows (exclude long text)
      const longLabels = new Set(["Preview", "Top evidence", "Ordering preview", "Summary", "Final reasoning"]);
      const sampleMetrics = stageRows(group.runs[0] || {});
      const shortMetrics = sampleMetrics.filter((m) => !longLabels.has(m.label));
      const longMetrics = sampleMetrics.filter((m) => longLabels.has(m.label));

      const columns = [
        { label: "Status", cell: (r) => `<span class="badge ${statusTone(r.current_stage_status || r.status)}">${escapeHtml((r.current_stage_status || r.status).replace(/_/g, " "))}</span>` },
        { label: "Cost", cell: (r) => escapeHtml(formatCurrency(r.usage?.cost_usd || 0)) },
        { label: "Issues", cell: (r) => String(r.quality_health?.issues_count || 0) },
        ...shortMetrics.map((m, idx) => ({
          label: m.label,
          cell: (r) => escapeHtml(stageRows(r)[idx]?.value || ""),
        })),
        { label: "Winner", cell: (r) => `
          <label class="checkbox-row">
            <input type="radio" name="${escapeHtml(runSelectionName(group))}" value="${escapeHtml(r.run_id)}" ${!ready ? "disabled" : ""} ${defaultSelection === r.run_id ? "checked" : ""}>
            <span>Advance</span>
          </label>
        ` },
        { label: "", cell: (r) => `<a href="/claims/${encodeURIComponent(group.claim_key)}?run_id=${encodeURIComponent(r.run_id)}" class="ghost-button small-button">Detail</a>` },
      ];

      const items = group.runs.map((r) => ({ ...r, _candidateColor: r.candidate_color }));

      const tableHtml = renderTransposedTable({
        items,
        columns,
        rowHeader: (r) => `
          <div class="candidate-chip">
            <span class="candidate-dot"></span>
            <strong>${escapeHtml(r.candidate_prefix || "R")}</strong>
            <span>${escapeHtml(r.candidate_label || r.provider_label || r.run_id)}</span>
          </div>
        `,
        detailContent: longMetrics.length ? (r) => {
          const runMetrics = stageRows(r);
          const longValues = runMetrics.filter((m) => longLabels.has(m.label));
          if (!longValues.length) return null;
          return longValues.map((m) => `<div class="detail-card"><span class="label">${escapeHtml(m.label)}</span><span class="value">${escapeHtml(m.value)}</span></div>`).join("");
        } : null,
        focusTest: (r) => isFocusedRun(r),
      });

      return `
        <section class="panel claim-section ${focused ? "focused-claim" : ""}" data-claim-group="${escapeHtml(group.claim_key)}">
          <div class="claim-section-header">
            <div class="stack">
              <h3 class="claim-title">${escapeHtml(group.text)}</h3>
              <div class="record-meta">
                <span>${group.runs.filter((run) => run.completed_stage === arenaData.current_stage).length}/${group.runs.length} complete</span>
                <span>${ready ? "Ready for winner selection" : "Waiting for candidates"}</span>
              </div>
            </div>
            <label class="checkbox-row">
              <input type="checkbox" data-skip-claim>
              <span>Skip claim</span>
            </label>
          </div>
          ${tableHtml}
        </section>
      `;
    }).join("");

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

  function claimTextForKey(claimKey) {
    return (arenaData.claim_groups || []).find((group) => group.claim_key === claimKey)?.text || claimKey;
  }

  function historyDecisionMarkup(entry, decision) {
    const claimText = claimTextForKey(decision.claim_key);
    if (decision.action !== "continue") {
      return `
        <article class="record-card">
          <strong>${escapeHtml(claimText)}</strong>
          <div class="record-meta">
            <span>Skipped</span>
            <span>No winner advanced from this stage</span>
          </div>
        </article>
      `;
    }

    const selectedRun = (entry.runs || []).find((run) => run.run_id === decision.selected_run_id);
    return `
      <article class="record-card" ${selectedRun ? `style="${candidateStyle({ color: selectedRun.candidate_color })}"` : ""}>
        <strong>${escapeHtml(claimText)}</strong>
        <div class="record-meta">
          <span>Winner advanced</span>
          ${selectedRun ? `<span>${escapeHtml(selectedRun.candidate_prefix || "R")} | ${escapeHtml(selectedRun.candidate_label || selectedRun.provider_label || selectedRun.run_id)}</span>` : `<span>${escapeHtml(decision.selected_run_id || "Unknown run")}</span>`}
        </div>
      </article>
    `;
  }

  function renderHistory() {
    const history = arenaData.stage_history || [];
    const target = byId("historyContent");
    if (!history.length) {
      target.innerHTML = `
        <div class="empty-state">
          <strong>No arena history yet.</strong>
          <span>Stage history appears here once the arena is continued or reopened.</span>
        </div>
      `;
      return;
    }
    target.innerHTML = history.map((entry) => {
      const promptSummary = promptSummaryForRuns(entry.runs || []);
      return `
        <article class="history-item">
          <div class="panel-header">
            <div>
              <h3 class="panel-title">${escapeHtml(entry.stage_label || stageLabel(entry.stage))}</h3>
              <p class="panel-subtitle">${escapeHtml(entry.source || "arena")}</p>
            </div>
            <div class="inline-actions">
              <span class="badge ${promptSummary.tone}">${escapeHtml(promptSummary.label)}</span>
              <span class="badge neutral-badge">${escapeHtml(formatDateTime(entry.created_at))}</span>
            </div>
          </div>
          <div class="record-meta">
            <span>${entry.runs?.length || 0} run${entry.runs?.length === 1 ? "" : "s"}</span>
            ${entry.continue_decisions?.length ? `<span>${entry.continue_decisions.filter((item) => item.action === "continue").length} advanced</span>` : ""}
            ${entry.continue_decisions?.length ? `<span>${entry.continue_decisions.filter((item) => item.action === "skip").length} skipped</span>` : ""}
          </div>
          <div class="pill-row">
            ${(entry.runs || []).map((run) => `
              <a class="pill" style="${candidateStyle({ color: run.candidate_color })}" href="/arena_results?arena_id=${encodeURIComponent(arenaData.arena_id)}&run_id=${encodeURIComponent(run.run_id)}">${escapeHtml(`${run.candidate_prefix || "R"} | ${run.candidate_label || run.provider_label || run.run_id}`)}</a>
            `).join("")}
          </div>
          ${entry.continue_decisions?.length ? `
            <div class="history-decision-list">
              ${entry.continue_decisions.map((decision) => historyDecisionMarkup(entry, decision)).join("")}
            </div>
          ` : ""}
        </article>
      `;
    }).join("");
  }

  function collectDecisions() {
    return (arenaData.claim_groups || []).map((group) => {
      const section = sectionForClaimKey(group.claim_key);
      const skipClaim = !!section?.querySelector("[data-skip-claim]")?.checked;
      return {
        claim_key: group.claim_key,
        skip_claim: skipClaim,
        selected_run_id: skipClaim ? null : (selectedRunIdForGroup(group) || null),
      };
    });
  }

  function renderContinueSummary() {
    const next = nextStage(arenaData.current_stage);
    if (!next) {
      byId("continuePanel").classList.add("hidden");
      return;
    }
    byId("continuePanel").classList.remove("hidden");

    if (!continuationPreflight) {
      byId("continueSummary").innerHTML = `
        <div class="summary-strip">
          <div class="summary-cell">
            <span class="label">Current Stage</span>
            <span class="value">${escapeHtml(stageLabel(arenaData.current_stage))}</span>
          </div>
          <div class="summary-cell">
            <span class="label">Next Stage</span>
            <span class="value">${escapeHtml(stageLabel(next))}</span>
          </div>
        </div>
        <p class="helper-text">Select a winner for each ready claim or skip the claim, then estimate the next stage before continuing.</p>
      `;
      byId("continueArenaBtn").disabled = true;
      return;
    }

    byId("continueSummary").innerHTML = `
      <div class="summary-strip">
        <div class="summary-cell">
          <span class="label">Claims Advancing</span>
          <span class="value">${continuationPreflight.totals.run_count}</span>
        </div>
        <div class="summary-cell">
          <span class="label">Expected Delta</span>
          <span class="value">${escapeHtml(formatCurrency(continuationPreflight.totals.expected_cost_usd))}</span>
        </div>
        <div class="summary-cell">
          <span class="label">Upper Bound</span>
          <span class="value">${escapeHtml(formatCurrency(continuationPreflight.totals.upper_bound_cost_usd))}</span>
        </div>
      </div>
      <div class="record-list">
        ${continuationPreflight.claims.map((claim) => `
          <article class="record-card">
            <strong>${escapeHtml(claim.text)}</strong>
            <div class="record-meta">
              <span>${escapeHtml(claim.action)}</span>
              ${claim.selected_run_id ? `<span>${escapeHtml(claim.selected_run_id)}</span>` : ""}
              ${claim.estimate ? `<span>Expected ${escapeHtml(formatCurrency(claim.estimate.expected?.cost_usd || 0))}</span>` : ""}
            </div>
          </article>
        `).join("")}
      </div>
    `;
    byId("continueArenaBtn").disabled = !continuationPreflight.totals.pricing_complete;
  }

  async function estimateContinuation() {
    continuationPreflight = await fetchJson(`/api/v1/arenas/${encodeURIComponent(config.arenaId)}/continue/preflight`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decisions: collectDecisions() }),
    });
    renderContinueSummary();
    renderStepper();
    return continuationPreflight;
  }

  function openContinueModal() {
    if (!continuationPreflight) {
      throw new Error("Estimate the next stage before continuing the arena.");
    }
    byId("continueModalBody").innerHTML = `
      <div class="summary-strip">
        <div class="summary-cell">
          <span class="label">Claims Advancing</span>
          <span class="value">${continuationPreflight.totals.run_count}</span>
        </div>
        <div class="summary-cell">
          <span class="label">Expected Delta</span>
          <span class="value">${escapeHtml(formatCurrency(continuationPreflight.totals.expected_cost_usd))}</span>
        </div>
        <div class="summary-cell">
          <span class="label">Upper Bound</span>
          <span class="value">${escapeHtml(formatCurrency(continuationPreflight.totals.upper_bound_cost_usd))}</span>
        </div>
      </div>
      <div class="record-list">
        ${continuationPreflight.claims.map((claim) => `
          <article class="record-card">
            <strong>${escapeHtml(claim.text)}</strong>
            <div class="record-meta">
              <span>${escapeHtml(claim.action)}</span>
              ${claim.source_run ? `<span>${escapeHtml(claim.source_run.candidate_prefix || "R")} | ${escapeHtml(claim.source_run.candidate_label || claim.source_run.provider_label || claim.source_run.run_id)}</span>` : ""}
            </div>
          </article>
        `).join("")}
      </div>
    `;
    byId("confirmContinueCheckbox").checked = false;
    byId("confirmContinueBtn").disabled = true;
    byId("continueModal").classList.remove("hidden");
  }

  async function continueArena() {
    const data = await fetchJson(`/api/v1/arenas/${encodeURIComponent(config.arenaId)}/continue`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        decisions: collectDecisions(),
        cost_confirmation: {
          accepted: true,
          expected_cost_usd: continuationPreflight.totals.expected_cost_usd,
          upper_bound_cost_usd: continuationPreflight.totals.upper_bound_cost_usd,
        },
      }),
    });
    window.location.href = `/arena_results?arena_id=${encodeURIComponent(data.arena_id)}`;
  }

  function activateTab(tabName) {
    document.querySelectorAll("[data-workspace-tab]").forEach((button) => {
      button.classList.toggle("active", button.dataset.workspaceTab === tabName);
    });
    byId("overviewPane").classList.toggle("hidden", tabName !== "overview");
    byId("claimsPane").classList.toggle("hidden", tabName !== "claims");
    byId("historyPane").classList.toggle("hidden", tabName !== "history");
  }

  function activeTabName() {
    return document.querySelector("[data-workspace-tab].active")?.dataset.workspaceTab || "overview";
  }

  function shouldAutoRefresh() {
    const status = arenaData?.summary?.status || arenaProgress?.summary?.status;
    return status === "in_progress";
  }

  function clearRefreshTimer() {
    if (refreshTimer) {
      window.clearTimeout(refreshTimer);
      refreshTimer = null;
    }
  }

  function updateWorkspaceMeta({ loading = false } = {}) {
    const stateBadge = byId("workspaceRefreshState");
    const lastUpdated = byId("workspaceLastUpdated");
    const refreshButton = byId("refreshWorkspaceBtn");
    if (!stateBadge || !lastUpdated) {
      return;
    }

    let label = "Snapshot";
    let badgeClass = "badge neutral-badge";
    if (loading) {
      label = lastRefreshCompletedAt ? "Refreshing" : "Loading";
      badgeClass = "badge neutral-badge";
    } else if (shouldAutoRefresh()) {
      if (document.hidden) {
        label = "Live paused";
        badgeClass = "badge warning-badge";
      } else {
        label = `Live every ${Math.round(autoRefreshMs / 1000)}s`;
        badgeClass = "badge success-badge";
      }
    }

    stateBadge.className = badgeClass;
    stateBadge.textContent = label;
    lastUpdated.textContent = lastRefreshCompletedAt
      ? `Last updated ${formatDateTime(lastRefreshCompletedAt)}`
      : "Syncing workspace...";
    if (refreshButton) {
      refreshButton.disabled = loading;
    }
  }

  function scheduleAutoRefresh() {
    clearRefreshTimer();
    if (!shouldAutoRefresh() || document.hidden) {
      updateWorkspaceMeta();
      return;
    }
    refreshTimer = window.setTimeout(() => {
      refreshWorkspace().catch(() => {});
    }, autoRefreshMs);
    updateWorkspaceMeta();
  }

  async function loadWorkspace({ initial = false } = {}) {
    const activeTab = initial ? "overview" : activeTabName();
    const [arena, progress] = await Promise.all([
      fetchJson(`/api/v1/arenas/${encodeURIComponent(config.arenaId)}`),
      fetchJson(`/api/v1/arenas/${encodeURIComponent(config.arenaId)}/progress`),
    ]);
    arenaData = arena;
    arenaProgress = progress;

    byId("arenaTitle").textContent = arena.summary?.title || arena.title || arena.arena_id;
    renderSummaryStrip(arena.summary || progress.summary);
    renderStepper();
    renderOverview();
    renderClaims();
    renderHistory();
    renderContinueSummary();
    lastRefreshCompletedAt = new Date().toISOString();

    const focused = focusRun();
    if (focused) {
      setStatus(byId("workspaceStatus"), {
        title: "Focused run highlighted",
        message: `${focused.candidate_prefix || "R"} for "${focused.text}" opened this workspace.`,
        tone: "info",
      });
    } else {
      hideStatus(byId("workspaceStatus"));
    }

    if (initial && !hasInitialized) {
      activateTab(focused ? (focused.status === "processed" ? "claims" : "overview") : "overview");
    } else {
      activateTab(activeTab);
    }
    hasInitialized = true;
  }

  async function refreshWorkspace({ initial = false, manual = false } = {}) {
    if (refreshInFlight) {
      return;
    }
    refreshInFlight = true;
    updateWorkspaceMeta({ loading: true });
    try {
      await loadWorkspace({ initial });
    } catch (error) {
      if (initial || manual) {
        setStatus(byId("workspaceStatus"), {
          title: initial ? "Arena workspace failed to load" : "Arena refresh failed",
          message: error.message,
          tone: "error",
        });
      } else {
        setStatus(byId("workspaceStatus"), {
          title: "Live update paused",
          message: error.message,
          tone: "warning",
        });
      }
    } finally {
      refreshInFlight = false;
      updateWorkspaceMeta();
      scheduleAutoRefresh();
    }
  }

  document.querySelectorAll("[data-workspace-tab]").forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.workspaceTab));
  });

  byId("claimGroups").addEventListener("change", () => {
    continuationPreflight = null;
    renderContinueSummary();
    renderStepper();
  });

  byId("estimateContinueBtn").addEventListener("click", () => {
    estimateContinuation().catch((error) => {
      activateTab("claims");
      setStatus(byId("workspaceStatus"), {
        title: "Select winners first",
        message: "Pick a winner for each claim below, or skip claims you don't want to advance.",
        tone: "warning",
      });
    });
  });

  byId("continueArenaBtn").addEventListener("click", () => {
    try {
      openContinueModal();
    } catch (error) {
      setStatus(byId("workspaceStatus"), {
        title: "Continuation blocked",
        message: error.message,
        tone: "error",
      });
    }
  });

  byId("cancelContinueBtn").addEventListener("click", () => {
    byId("continueModal").classList.add("hidden");
  });

  byId("confirmContinueCheckbox").addEventListener("change", (event) => {
    byId("confirmContinueBtn").disabled = !event.target.checked;
  });

  byId("confirmContinueBtn").addEventListener("click", () => {
    continueArena().catch((error) => {
      setStatus(byId("workspaceStatus"), {
        title: "Continuation failed",
        message: error.message,
        tone: "error",
      });
    });
  });

  byId("refreshWorkspaceBtn").addEventListener("click", () => {
    refreshWorkspace({ manual: true }).catch(() => {});
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      clearRefreshTimer();
      updateWorkspaceMeta();
      return;
    }
    if (shouldAutoRefresh()) {
      refreshWorkspace().catch(() => {});
    } else {
      updateWorkspaceMeta();
    }
  });

  window.addEventListener("beforeunload", clearRefreshTimer);

  hideStatus(byId("workspaceStatus"));
  updateWorkspaceMeta({ loading: true });
  refreshWorkspace({ initial: true }).catch(() => {});
})();
