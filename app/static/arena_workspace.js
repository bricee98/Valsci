(() => {
  const {
    escapeHtml,
    fetchJson,
    formatCurrency,
    formatDateTime,
    formatDurationMs,
    formatTokens,
    hideStatus,
    setStatus,
    flashButton,
    buttonBusy,
    stageLabel,
    ratingLabel,
    ratingTone,
    ratingChip,
    candidateStyle,
    renderTransposedTable,
  } = window.ValsciUI;

  const config = window.arenaWorkspaceConfig || {};
  const byId = (id) => document.getElementById(id);
  const orderedStages = ["query_generation", "paper_analysis", "venue_scoring", "final_report"];

  let arenaData = null;
  let arenaProgress = null;
  let candidateStats = [];
  let continuationPreflight = null;
  let hasInitialized = false;
  let refreshTimer = null;
  let refreshInFlight = false;
  let lastRefreshCompletedAt = null;
  let matrixSort = "default";
  const expandedCompare = new Set();
  const advanceSelections = {};
  const skipSelections = {};

  const autoRefreshMs = 5000;

  // ===== Generic helpers =====

  function nextStage(stage) {
    const index = orderedStages.indexOf(stage);
    if (index < 0 || index >= orderedStages.length - 1) {
      return null;
    }
    return orderedStages[index + 1];
  }

  function isTerminalStage() {
    return !nextStage(arenaData?.current_stage);
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

  // ===== Run / candidate helpers =====

  function isRunFailed(run) {
    const status = String(run?.status || "").toLowerCase();
    const stageStatus = String(run?.current_stage_status || "").toLowerCase();
    // A run can fail terminally yet be marked "processed" (e.g. query generation
    // gave up) — evaluation_failed flags those so they read as failed, not a verdict.
    return run?.evaluation_failed === true
      || ["error", "failed"].includes(status)
      || ["error", "failed"].includes(stageStatus);
  }

  function runResolved(run) {
    return run.completed_stage === arenaData.current_stage || isRunFailed(run);
  }

  function runReport(run) {
    return run?.claim_data?.report || run?.report || {};
  }

  function runRating(run) {
    const direct = run?.claimRating;
    if (direct !== null && direct !== undefined) {
      return Number(direct);
    }
    const fromData = runReport(run)?.claimRating;
    if (fromData === null || fromData === undefined) {
      return null;
    }
    return Number(fromData);
  }

  // A real verdict is 0–5 (0 = No Evidence). The Unrated sentinel (-1, or a
  // null/undefined rating on a completed run) is NOT a verdict and must never be
  // averaged into the mean or counted toward agreement.
  function isScoredVerdict(rating) {
    return rating !== null && rating !== undefined && Number.isFinite(rating) && rating >= 0;
  }

  // The model a candidate actually ran. A candidate's provider default_model can
  // differ from the per-stage models it was configured to use (model_overrides),
  // so prefer the models actually used across stages. Returns the single model
  // when uniform, "Mixed" when stages used different models, else default_model.
  function effectiveModelLabel(source) {
    if (!source) {
      return "";
    }
    const taskModels = source.effective_models?.task_models;
    let models = taskModels && typeof taskModels === "object" ? Object.values(taskModels) : [];
    if (!models.length && source.model_overrides && typeof source.model_overrides === "object") {
      models = Object.values(source.model_overrides);
    }
    const unique = [...new Set(models.filter(Boolean))];
    if (unique.length === 1) {
      return unique[0];
    }
    if (unique.length > 1) {
      return "Mixed";
    }
    return source.default_model || "";
  }

  function runTokens(run) {
    return Number(run?.usage?.total_tokens || 0) || Number(run?.trace_summary?.total_tokens || 0);
  }

  function candidateKeyForRun(run) {
    return run?.candidate_id || `${run?.candidate_prefix || "?"}::${run?.provider_id || "unknown"}`;
  }

  function candidateName(run) {
    return `${run?.candidate_prefix || "R"} · ${run?.candidate_label || run?.provider_label || run?.run_id || "Unknown"}`;
  }

  function candidateColumns() {
    const fromProgress = (arenaProgress?.candidates || []).map((entry) => ({
      key: entry.candidate_id,
      prefix: entry.candidate?.prefix || "?",
      label: entry.candidate?.label || entry.provider_label || entry.candidate_id,
      color: entry.candidate?.color,
      model: effectiveModelLabel(entry),
      providerLabel: entry.provider_label || entry.provider_id || "",
      index: entry.candidate?.index ?? 0,
    }));
    if (fromProgress.length) {
      return fromProgress;
    }
    const seen = new Map();
    for (const group of arenaData?.claim_groups || []) {
      for (const run of group.runs || []) {
        const key = candidateKeyForRun(run);
        if (!seen.has(key)) {
          seen.set(key, {
            key,
            prefix: run.candidate_prefix || "?",
            label: run.candidate_label || run.provider_label || key,
            color: run.candidate_color,
            model: effectiveModelLabel(run),
            providerLabel: run.provider_label || "",
            index: run.candidate_index ?? seen.size,
          });
        }
      }
    }
    return Array.from(seen.values()).sort((a, b) => a.index - b.index);
  }

  function runsByCandidate(group) {
    const map = new Map();
    (group.runs || []).forEach((run) => {
      map.set(candidateKeyForRun(run), run);
    });
    return map;
  }

  function orderedRuns(group) {
    return [...(group.runs || [])].sort((a, b) => (a.candidate_index ?? 0) - (b.candidate_index ?? 0));
  }

  function findRunById(runId) {
    for (const group of arenaData?.claim_groups || []) {
      const run = (group.runs || []).find((item) => item.run_id === runId);
      if (run) {
        return run;
      }
    }
    return null;
  }

  function preferenceForClaim(claimKey) {
    return (arenaData?.preferences || {})[claimKey] || null;
  }

  function isPreferredRun(group, run) {
    return preferenceForClaim(group.claim_key)?.run_id === run.run_id;
  }

  function paperKey(entry) {
    const paper = entry?.paper || {};
    const corpusId = paper.corpus_id ?? paper.corpusId;
    if (corpusId !== null && corpusId !== undefined && String(corpusId).trim() !== "") {
      return `corpus:${corpusId}`;
    }
    const title = String(paper.title || "").trim().toLowerCase();
    return title ? `title:${title}` : null;
  }

  function paperSet(run) {
    const keys = (run?.claim_data?.processed_papers || []).map(paperKey).filter(Boolean);
    return new Set(keys);
  }

  function jaccard(setA, setB) {
    if (!setA.size && !setB.size) {
      return 0;
    }
    let intersection = 0;
    setA.forEach((value) => {
      if (setB.has(value)) {
        intersection += 1;
      }
    });
    return intersection / (setA.size + setB.size - intersection);
  }

  // ===== Group-level analysis =====

  function groupRatings(group) {
    return (group.runs || [])
      .map(runRating)
      .filter(isScoredVerdict);
  }

  function groupAgreement(group) {
    // Classify every run that reached a terminal state. A completed run with no
    // scored verdict (-1/null) is an "unrated" abstention — not a verdict to
    // agree or disagree with, but it shouldn't read as still-pending either.
    const verdicts = [];   // 1–5
    let noEvidence = 0;    // rating 0
    let unrated = 0;       // resolved but no scored verdict
    (group.runs || []).forEach((run) => {
      if (isRunFailed(run)) {
        return;
      }
      const rating = runRating(run);
      if (isScoredVerdict(rating)) {
        if (rating === 0) {
          noEvidence += 1;
        } else {
          verdicts.push(rating);
        }
      } else if (runResolved(run)) {
        unrated += 1;
      }
    });

    const decided = verdicts.length + noEvidence + unrated;
    if (decided < 2) {
      return { kind: "pending", label: decided ? "Needs 2+ verdicts" : "Pending", sortValue: -1 };
    }
    // A model that found evidence (1–5) sitting next to one that abstained
    // (No Evidence or Unrated) is a disagreement about whether evidence exists.
    if (verdicts.length && (noEvidence || unrated)) {
      return { kind: "split", label: "Split: evidence vs. none", sortValue: 90 };
    }
    // No scored verdicts at all — every model abstained. "No Evidence" (0) is a
    // real shared verdict, so concurring on it is agreement; an Unrated abstention
    // is not a verdict, so that reads neutral rather than green.
    if (!verdicts.length) {
      if (unrated) {
        return { kind: "muted", label: noEvidence ? "No verdict" : "Both unrated", sortValue: 1 };
      }
      return { kind: "agree", label: "Agree: no evidence", sortValue: 0 };
    }
    const spread = Math.max(...verdicts) - Math.min(...verdicts);
    if (spread === 0) {
      return { kind: "agree", label: "Agree", sortValue: 0 };
    }
    return { kind: "diff", label: `Disagree by ${spread} pt${spread > 1 ? "s" : ""}`, sortValue: spread };
  }

  function agreementBadge(agreement) {
    const tone = agreement.kind === "agree"
      ? "success-badge"
      : (agreement.kind === "split" || agreement.sortValue >= 2)
        ? "warning-badge"
        : "neutral-badge";
    return `<span class="badge ${tone}">${escapeHtml(agreement.label)}</span>`;
  }

  function groupMeanRating(group) {
    const rated = groupRatings(group).filter((value) => value > 0);
    if (!rated.length) {
      return null;
    }
    return rated.reduce((sum, value) => sum + value, 0) / rated.length;
  }

  function sortedGroups() {
    const groups = [...(arenaData?.claim_groups || [])];
    if (matrixSort === "disagreement") {
      groups.sort((a, b) => groupAgreement(b).sortValue - groupAgreement(a).sortValue);
    } else if (matrixSort === "rating") {
      groups.sort((a, b) => {
        const meanA = groupMeanRating(a);
        const meanB = groupMeanRating(b);
        return (meanA === null ? Infinity : meanA) - (meanB === null ? Infinity : meanB);
      });
    }
    return groups;
  }

  function claimReadyForDecision(group) {
    const runs = group.runs || [];
    if (!runs.length) {
      return false;
    }
    const completed = runs.filter((run) => run.completed_stage === arenaData.current_stage);
    return runs.every(runResolved) && completed.length > 0;
  }

  function totalFailedRuns() {
    return (arenaData?.claim_groups || []).reduce(
      (count, group) => count + (group.runs || []).filter(isRunFailed).length,
      0
    );
  }

  // ===== Candidate scorecard stats =====

  function computeCandidateStats() {
    const candidates = candidateColumns();
    const groups = arenaData?.claim_groups || [];
    const stats = new Map();
    candidates.forEach((candidate) => {
      stats.set(candidate.key, {
        ...candidate,
        ratedValues: [],
        dist: {},
        noEvidence: 0,
        agree: 0,
        agreeTotal: 0,
        picks: 0,
        failed: 0,
        evidenceClaims: 0,
        papersTotal: 0,
        tokens: 0,
        cost: 0,
        elapsedMs: 0,
        issues: 0,
        overlapSum: 0,
        overlapCount: 0,
        runCount: 0,
      });
    });

    groups.forEach((group) => {
      const byKey = runsByCandidate(group);
      const ratings = groupRatings(group);
      const counts = {};
      ratings.forEach((value) => {
        counts[value] = (counts[value] || 0) + 1;
      });
      let majority = null;
      let bestCount = 0;
      let tie = false;
      Object.entries(counts).forEach(([value, count]) => {
        if (count > bestCount) {
          bestCount = count;
          majority = Number(value);
          tie = false;
        } else if (count === bestCount) {
          tie = true;
        }
      });
      if (bestCount < 2 || tie) {
        majority = null;
      }

      const candidateSets = candidates.map((candidate) => ({
        key: candidate.key,
        set: paperSet(byKey.get(candidate.key)),
      }));

      candidates.forEach((candidate) => {
        const run = byKey.get(candidate.key);
        if (!run) {
          return;
        }
        const entry = stats.get(candidate.key);
        entry.runCount += 1;
        const rating = runRating(run);
        if (isScoredVerdict(rating)) {
          entry.dist[rating] = (entry.dist[rating] || 0) + 1;
          if (rating === 0) {
            entry.noEvidence += 1;
          } else {
            entry.ratedValues.push(rating);
          }
          if (majority !== null) {
            entry.agreeTotal += 1;
            if (rating === majority) {
              entry.agree += 1;
            }
          }
        }
        if (isRunFailed(run)) {
          entry.failed += 1;
        }
        const papers = run.claim_data?.processed_papers || [];
        if (papers.length) {
          entry.evidenceClaims += 1;
        }
        entry.papersTotal += papers.length;
        entry.tokens += runTokens(run);
        entry.cost += Number(run.usage?.cost_usd || 0);
        entry.elapsedMs += Number(run.total_elapsed_ms || 0);
        entry.issues += Number(run.quality_health?.issues_count || 0);
        if (isPreferredRun(group, run)) {
          entry.picks += 1;
        }
        const mySet = paperSet(run);
        if (mySet.size) {
          candidateSets.forEach((other) => {
            if (other.key !== candidate.key && other.set.size) {
              entry.overlapSum += jaccard(mySet, other.set);
              entry.overlapCount += 1;
            }
          });
        }
      });
    });

    return candidates.map((candidate) => {
      const entry = stats.get(candidate.key);
      const ratedCount = entry.ratedValues.length;
      return {
        ...entry,
        meanRating: ratedCount ? entry.ratedValues.reduce((sum, value) => sum + value, 0) / ratedCount : null,
        ratedCount,
        agreementPct: entry.agreeTotal ? entry.agree / entry.agreeTotal : null,
        overlapPct: entry.overlapCount ? entry.overlapSum / entry.overlapCount : null,
      };
    });
  }

  // ===== Summary strip + stepper =====

  function renderSummaryStrip(summary) {
    const failed = totalFailedRuns();
    byId("arenaSummaryStrip").innerHTML = `
      <div class="summary-cell">
        <span class="label">Status</span>
        <span class="value">${escapeHtml(String(summary.status || "unknown").replace(/_/g, " "))}</span>
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
        <span class="label">Models</span>
        <span class="value">${summary.candidate_count}</span>
      </div>
      ${failed ? `
        <div class="summary-cell">
          <span class="label">Failed Runs</span>
          <span class="value"><span class="badge error-badge">${failed}</span></span>
        </div>
      ` : ""}
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

  function isStagedArena() {
    const history = arenaData?.stage_history || [];
    return arenaData?.current_stage !== "final_report"
      || history.some((entry) => (entry.stage || "final_report") !== "final_report");
  }

  function allRunsResolved() {
    const groups = arenaData?.claim_groups || [];
    return groups.length > 0 && groups.every((group) => (group.runs || []).length && (group.runs || []).every(runResolved));
  }

  function stepState(key) {
    const currentStage = arenaData.current_stage;
    const currentIndex = orderedStages.indexOf(currentStage);
    const status = arenaData.summary?.status || "in_progress";

    if (key === "setup") {
      return "complete";
    }
    if (key === "running") {
      if (status !== "in_progress" || allRunsResolved()) {
        return "complete";
      }
      return "active";
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
    if (key === "compare") {
      return allRunsResolved() ? "active" : "";
    }
    if (key === "select_winners") {
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
    let items;
    if (!isStagedArena()) {
      items = [
        { key: "setup", label: "Setup" },
        { key: "running", label: "Running" },
        { key: "compare", label: "Compare Results" },
      ];
    } else if (isTerminalStage()) {
      items = [
        { key: "setup", label: "Setup" },
        { key: "running", label: "Running" },
        { key: "query_generation", label: "Review Queries" },
        { key: "paper_analysis", label: "Review Evidence" },
        { key: "venue_scoring", label: "Review Scores" },
        { key: "final_report", label: "Final Reports" },
        { key: "compare", label: "Compare Results" },
      ];
    } else {
      items = [
        { key: "setup", label: "Setup" },
        { key: "running", label: "Running" },
        { key: "query_generation", label: "Review Queries" },
        { key: "paper_analysis", label: "Review Evidence" },
        { key: "venue_scoring", label: "Review Scores" },
        { key: "final_report", label: "Final Reports" },
        { key: "select_winners", label: "Select Winners" },
        { key: "continue", label: "Continue" },
      ];
    }
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

  // ===== Overview: scorecard + operations =====

  function candidateRowHeader(entry) {
    return `
      <div class="candidate-chip">
        <span class="candidate-dot"></span>
        <strong>${escapeHtml(entry.prefix || "?")}</strong>
        <span>${escapeHtml(entry.label || entry.key)}</span>
      </div>
    `;
  }

  function renderScorecardTable() {
    const stats = candidateStats;
    if (!stats.length) {
      return "";
    }
    const maxAgreement = Math.max(...stats.map((entry) => entry.agreementPct ?? -1));
    const maxPicks = Math.max(...stats.map((entry) => entry.picks));
    const positiveCosts = stats.map((entry) => entry.cost).filter((value) => value > 0);
    const minCost = positiveCosts.length ? Math.min(...positiveCosts) : null;
    const positiveTimes = stats.map((entry) => entry.elapsedMs).filter((value) => value > 0);
    const minTime = positiveTimes.length ? Math.min(...positiveTimes) : null;
    const maxEvidence = Math.max(...stats.map((entry) => entry.evidenceClaims));

    const columns = [
      {
        label: "Mean Verdict",
        cell: (entry) => entry.meanRating !== null
          ? `<strong>${entry.meanRating.toFixed(1)}</strong> <span class="helper-text">(${entry.ratedCount} rated, 1–5)</span>`
          : `<span class="helper-text">—</span>`,
      },
      {
        label: "Verdict Mix",
        cell: (entry) => {
          const chips = [5, 4, 3, 2, 1, 0]
            .filter((rating) => entry.dist[rating])
            .map((rating) => `<span class="rating-chip dist-chip ${ratingTone(rating)}" title="${entry.dist[rating]}× ${escapeHtml(ratingLabel(rating))}">${entry.dist[rating]}× ${rating}</span>`)
            .join("");
          return chips ? `<div class="dist-row">${chips}</div>` : `<span class="helper-text">No verdicts yet</span>`;
        },
      },
      {
        label: "No Evidence",
        cell: (entry) => String(entry.noEvidence),
      },
      {
        label: "Majority Agreement",
        cell: (entry) => entry.agreementPct !== null
          ? `${Math.round(entry.agreementPct * 100)}% <span class="helper-text">(${entry.agree}/${entry.agreeTotal})</span>`
          : `<span class="helper-text">—</span>`,
        highlight: (entry) => entry.agreementPct !== null && entry.agreementPct === maxAgreement && maxAgreement >= 0,
      },
      {
        label: "Your Picks",
        cell: (entry) => String(entry.picks),
        highlight: (entry) => entry.picks > 0 && entry.picks === maxPicks,
      },
      {
        label: "Failed Runs",
        cell: (entry) => entry.failed ? `<span class="badge error-badge">${entry.failed}</span>` : "0",
      },
      {
        label: "Claims w/ Evidence",
        cell: (entry) => `${entry.evidenceClaims} <span class="helper-text">(${entry.papersTotal} papers)</span>`,
        highlight: (entry) => entry.evidenceClaims > 0 && entry.evidenceClaims === maxEvidence,
      },
      {
        label: "Retrieval Overlap",
        cell: (entry) => entry.overlapPct !== null
          ? `${Math.round(entry.overlapPct * 100)}%`
          : `<span class="helper-text">—</span>`,
      },
      {
        label: "Tokens",
        cell: (entry) => escapeHtml(formatTokens(entry.tokens)),
      },
      {
        label: "Actual Cost",
        cell: (entry) => escapeHtml(formatCurrency(entry.cost)),
        highlight: (entry) => minCost !== null && entry.cost === minCost,
      },
      {
        label: "Total Time",
        cell: (entry) => escapeHtml(formatDurationMs(entry.elapsedMs)),
        highlight: (entry) => minTime !== null && entry.elapsedMs === minTime,
      },
    ];

    const items = stats.map((entry) => ({ ...entry, _candidateColor: entry.color }));
    return renderTransposedTable({
      items,
      columns,
      rowHeader: candidateRowHeader,
    });
  }

  function renderOperationsTable() {
    const candidates = arenaProgress?.candidates || [];
    if (!candidates.length) {
      return "";
    }
    const positiveTimes = candidates.map((c) => Number(c.total_elapsed_ms || 0)).filter((v) => v > 0);
    const fastest = positiveTimes.length ? Math.min(...positiveTimes) : null;
    const fewestIssues = Math.min(...candidates.map((c) => Number(c.issue_count || 0)));

    const columns = [
      { label: "Provider", cell: (c) => escapeHtml(c.provider_label || c.provider_id || "Unknown") },
      { label: "Model", cell: (c) => escapeHtml(effectiveModelLabel(c) || "Unknown") },
      { label: "Status", cell: (c) => `<span class="badge ${statusTone(c.status)}">${escapeHtml(String(c.status || "unknown").replace(/_/g, " "))}</span>` },
      { label: "Stage", cell: (c) => escapeHtml(stageLabel(c.current_stage)) },
      { label: "Elapsed", cell: (c) => escapeHtml(formatDurationMs(c.total_elapsed_ms)), highlight: (c) => fastest !== null && Number(c.total_elapsed_ms || 0) === fastest },
      { label: "Query Time", cell: (c) => escapeHtml(formatDurationMs(c.stage_timings_ms?.query_generation)) },
      { label: "Evidence Time", cell: (c) => escapeHtml(formatDurationMs(c.stage_timings_ms?.paper_analysis)) },
      { label: "Scoring Time", cell: (c) => escapeHtml(formatDurationMs(c.stage_timings_ms?.venue_scoring)) },
      { label: "Report Time", cell: (c) => escapeHtml(formatDurationMs(c.stage_timings_ms?.final_report)) },
      { label: "Issues", cell: (c) => String(c.issue_count || 0), highlight: (c) => Number(c.issue_count || 0) === fewestIssues },
      { label: "Retries", cell: (c) => String(c.retry_count || 0) },
      { label: "Expected Cost", cell: (c) => escapeHtml(formatCurrency(c.expected_cost_usd)) },
    ];

    const items = candidates.map((c) => ({ ...c, _candidateColor: c.candidate?.color }));
    const focused = focusRun();
    return renderTransposedTable({
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
    });
  }

  function renderOverview() {
    const overview = byId("overviewContent");
    if (!(arenaProgress?.candidates || []).length) {
      overview.innerHTML = `<div class="empty-state"><strong>No candidate progress yet.</strong><span>Launch runs from the arena builder.</span></div>`;
      return;
    }
    const focused = focusRun();
    overview.innerHTML = `
      ${focused ? `
        <div class="status-card info-card">
          <strong>Focused run highlighted</strong>
          <span>${escapeHtml(focused.candidate_prefix || "R")} for claim "${escapeHtml(focused.text)}" is highlighted.</span>
        </div>
      ` : ""}
      <div>
        <h3 class="panel-title">Model Scorecard</h3>
        <p class="panel-subtitle">Verdict quality, agreement, and cost per model. Open the Claims tab to compare outputs side by side and pick the best one per claim.</p>
      </div>
      ${renderScorecardTable()}
      <div style="margin-top:10px">
        <h3 class="panel-title">Pipeline Operations</h3>
        <p class="panel-subtitle">Runtime, reliability, and spend for each model's pipeline runs.</p>
      </div>
      ${renderOperationsTable()}
    `;
  }

  // ===== Verdict matrix =====

  function matrixCell(group, run) {
    if (!run) {
      return `<span class="helper-text">—</span>`;
    }
    if (isRunFailed(run)) {
      return `<span class="rating-chip rating-failed" title="Run failed (${escapeHtml(run.status || "error")})">✕ Failed</span>`;
    }
    const rating = runRating(run);
    if (rating !== null && Number.isFinite(rating)) {
      const star = isPreferredRun(group, run)
        ? ` <span class="pick-star" title="Your pick for this claim">★</span>`
        : "";
      return `${ratingChip(rating, { label: run.rating_label !== "Unrated" ? run.rating_label : undefined })}${star}`;
    }
    if (run.completed_stage === arenaData.current_stage) {
      return `<span class="rating-chip rating-none" title="Stage checkpoint complete">Ready</span>`;
    }
    const statusText = String(run.current_stage_status || run.status || "queued").replace(/_/g, " ");
    return `<span class="rating-chip rating-none" title="${escapeHtml(statusText)}">${escapeHtml(stageLabel(run.current_stage))}…</span>`;
  }

  function renderMatrix() {
    const target = byId("verdictMatrix");
    if (!target) {
      return;
    }
    const groups = sortedGroups();
    const candidates = candidateColumns();
    if (!groups.length || !candidates.length) {
      target.innerHTML = "";
      return;
    }

    const statsByKey = new Map(candidateStats.map((entry) => [entry.key, entry]));
    const headCells = candidates.map((candidate) => `
      <th title="${escapeHtml([candidate.providerLabel, candidate.model].filter(Boolean).join(" · "))}">
        <div class="candidate-chip" style="${candidateStyle({ color: candidate.color })}">
          <span class="candidate-dot"></span>
          <strong>${escapeHtml(candidate.prefix)}</strong>
          <span>${escapeHtml(candidate.label || candidate.model)}</span>
        </div>
      </th>
    `).join("");

    const bodyRows = groups.map((group) => {
      const byKey = runsByCandidate(group);
      const agreement = groupAgreement(group);
      const cells = candidates.map((candidate) => `<td>${matrixCell(group, byKey.get(candidate.key))}</td>`).join("");
      return `
        <tr>
          <th>
            <button type="button" class="matrix-claim-btn" data-claim-key="${escapeHtml(group.claim_key)}" title="${escapeHtml(group.text)} — jump to side-by-side outputs">
              ${escapeHtml(group.text)}
            </button>
          </th>
          ${cells}
          <td>${agreementBadge(agreement)}</td>
        </tr>
      `;
    }).join("");

    const footCells = candidates.map((candidate) => {
      const entry = statsByKey.get(candidate.key);
      return `<td>${entry && entry.meanRating !== null ? entry.meanRating.toFixed(1) : "—"}</td>`;
    }).join("");

    target.innerHTML = `
      <div class="verdict-matrix-wrap">
        <table class="verdict-matrix">
          <thead>
            <tr>
              <th>Claim</th>
              ${headCells}
              <th>Agreement</th>
            </tr>
          </thead>
          <tbody>${bodyRows}</tbody>
          <tfoot>
            <tr>
              <th>Mean verdict (1–5)</th>
              ${footCells}
              <td></td>
            </tr>
          </tfoot>
        </table>
      </div>
    `;

    target.querySelectorAll(".matrix-claim-btn").forEach((button) => {
      button.addEventListener("click", () => {
        const claimKey = button.dataset.claimKey;
        if (!expandedCompare.has(claimKey)) {
          expandedCompare.add(claimKey);
          renderClaims();
        }
        const section = sectionForClaimKey(claimKey);
        section?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
  }

  // ===== Side-by-side output comparison =====

  function comparePaperList(run, allRuns, { showScores = false } = {}) {
    const papers = run?.claim_data?.processed_papers || [];
    if (!papers.length) {
      return null;
    }
    const presence = new Map();
    allRuns.forEach((other) => {
      (other?.claim_data?.processed_papers || []).forEach((entry) => {
        const key = paperKey(entry);
        if (!key) {
          return;
        }
        if (!presence.has(key)) {
          presence.set(key, new Set());
        }
        presence.get(key).add(other.candidate_prefix || "?");
      });
    });

    const sorted = showScores
      ? [...papers].sort((a, b) => Number(b?.score ?? -1) - Number(a?.score ?? -1))
      : papers;

    return sorted.map((entry) => {
      const paper = entry?.paper || {};
      const key = paperKey(entry);
      const foundBy = key ? presence.get(key) : null;
      const others = foundBy
        ? [...foundBy].filter((prefix) => prefix !== (run.candidate_prefix || "?")).sort()
        : [];
      const mark = foundBy
        ? (others.length
          ? `<span class="compare-paper-mark">Also found by ${escapeHtml(others.join(", "))}</span>`
          : `<span class="compare-paper-mark unique">Unique to ${escapeHtml(run.candidate_prefix || "?")}</span>`)
        : "";
      const url = typeof paper.url === "string" && /^https?:\/\//i.test(paper.url) ? paper.url : null;
      const title = escapeHtml(paper.title || "Untitled");
      const scoreValue = entry?.score;
      const hasScore = scoreValue !== undefined && scoreValue !== null && scoreValue !== -1;
      const meta = [
        paper.year ? escapeHtml(String(paper.year)) : "",
        entry?.relevance !== undefined && entry?.relevance !== null ? `Relevance ${escapeHtml(String(entry.relevance))}` : "",
        showScores && hasScore ? `Score ${escapeHtml(String(scoreValue))}` : "",
        (entry?.excerpts || []).length ? `${(entry.excerpts || []).length} excerpt${(entry.excerpts || []).length === 1 ? "" : "s"}` : "",
      ].filter(Boolean).join(" · ");
      return `
        <div class="compare-paper">
          <span class="compare-paper-title">${url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${title}</a>` : title}</span>
          <span class="compare-paper-meta">${meta}${mark ? ` ${mark}` : ""}</span>
        </div>
      `;
    }).join("");
  }

  function compareQueryList(run) {
    const queries = run?.claim_data?.semantic_scholar_queries || [];
    if (!queries.length) {
      return null;
    }
    return `<ol class="compare-query-list">${queries.map((query) => `<li>${escapeHtml(query)}</li>`).join("")}</ol>`;
  }

  function compareSections(stage) {
    if (stage === "query_generation") {
      return [
        { label: "Search Queries", render: (run) => compareQueryList(run) },
      ];
    }
    if (stage === "paper_analysis") {
      return [
        { label: "Relevant Papers", render: (run, allRuns) => comparePaperList(run, allRuns) },
        { label: "Search Queries", render: (run) => compareQueryList(run) },
      ];
    }
    if (stage === "venue_scoring") {
      return [
        { label: "Scored Papers", render: (run, allRuns) => comparePaperList(run, allRuns, { showScores: true }) },
      ];
    }
    return [
      {
        label: "Verdict",
        render: (run) => {
          const rating = runRating(run);
          if (rating === null || !Number.isFinite(rating)) {
            return null;
          }
          return ratingChip(rating, { label: run.rating_label !== "Unrated" ? run.rating_label : undefined });
        },
      },
      { label: "Summary", render: (run) => { const text = runReport(run)?.explanation; return text ? escapeHtml(text) : null; } },
      { label: "Final Reasoning", render: (run) => { const text = runReport(run)?.finalReasoning; return text ? escapeHtml(text) : null; } },
      { label: "Evidence Used", render: (run, allRuns) => comparePaperList(run, allRuns, { showScores: true }) },
      { label: "Search Queries", render: (run) => compareQueryList(run) },
    ];
  }

  function renderCompareGrid(group) {
    const runs = orderedRuns(group);
    if (!runs.length) {
      return "";
    }
    const sections = compareSections(arenaData.current_stage);
    const gridStyle = `grid-template-columns: 140px repeat(${runs.length}, minmax(280px, 1fr));`;

    const headerCells = runs.map((run) => {
      const failed = isRunFailed(run);
      const pick = isPreferredRun(group, run);
      return `
        <div class="compare-cell compare-head" style="${candidateStyle({ color: run.candidate_color })}">
          <div class="candidate-chip">
            <span class="candidate-dot"></span>
            <strong>${escapeHtml(run.candidate_prefix || "R")}</strong>
            <span>${escapeHtml(run.candidate_label || run.provider_label || run.run_id)}</span>
          </div>
          <div class="record-meta">
            <span>${escapeHtml(effectiveModelLabel(run))}</span>
            <span class="badge ${statusTone(run.current_stage_status || run.status)}">${escapeHtml(String(run.current_stage_status || run.status || "").replace(/_/g, " "))}</span>
            ${pick ? `<span class="badge success-badge">★ Your pick</span>` : ""}
          </div>
          <div class="inline-actions">
            <a href="/claims/${encodeURIComponent(group.claim_key)}?run_id=${encodeURIComponent(run.run_id)}" class="ghost-button small-button">Detail</a>
            ${failed ? `<button type="button" class="secondary-button small-button" data-retry-run="${escapeHtml(run.run_id)}">Retry</button>` : ""}
          </div>
        </div>
      `;
    }).join("");

    const sectionRows = sections.map((section) => {
      const cells = runs.map((run) => {
        if (isRunFailed(run)) {
          return `<div class="compare-cell compare-muted">Run failed — no output. Use Retry to resume it.</div>`;
        }
        const content = section.render(run, runs);
        return content
          ? `<div class="compare-cell">${content}</div>`
          : `<div class="compare-cell compare-muted">Not available yet</div>`;
      }).join("");
      return `
        <div class="compare-label">${escapeHtml(section.label)}</div>
        ${cells}
      `;
    }).join("");

    return `
      <div class="compare-grid-wrap">
        <div class="compare-grid" style="${gridStyle}">
          <div class="compare-label">Candidate</div>
          ${headerCells}
          ${sectionRows}
        </div>
      </div>
    `;
  }

  // ===== Claims tab =====

  function runSelectionName(group) {
    return `winner-${group.claim_key}`;
  }

  function sectionForClaimKey(claimKey) {
    return Array.from(document.querySelectorAll("[data-claim-group]")).find((section) => section.dataset.claimGroup === claimKey) || null;
  }

  function selectedRunIdForGroup(group) {
    const section = sectionForClaimKey(group.claim_key);
    if (!section) {
      return advanceSelections[group.claim_key] || null;
    }
    const selected = section.querySelector(`input[name="${runSelectionName(group)}"]:checked`);
    return selected?.value || null;
  }

  function stageMetricColumns(stage) {
    if (stage === "query_generation") {
      return [
        { label: "Queries", cell: (r) => String((r.claim_data?.semantic_scholar_queries || []).length) },
        { label: "Distinct", cell: (r) => String(new Set(r.claim_data?.semantic_scholar_queries || []).size) },
      ];
    }
    if (stage === "paper_analysis") {
      return [
        { label: "Relevant", cell: (r) => String((r.claim_data?.processed_papers || []).length) },
        { label: "Inaccessible", cell: (r) => String((r.claim_data?.inaccessible_papers || []).length) },
      ];
    }
    if (stage === "venue_scoring") {
      return [
        {
          label: "Scored",
          cell: (r) => String((r.claim_data?.processed_papers || []).filter((paper) => paper?.score !== undefined && paper?.score !== null && paper?.score !== -1).length),
        },
      ];
    }
    return [
      { label: "Papers", cell: (r) => String((r.claim_data?.processed_papers || []).length) },
    ];
  }

  function renderClaims() {
    const target = byId("claimGroups");
    const groups = sortedGroups();
    if (!groups.length) {
      target.innerHTML = `<div class="empty-state"><strong>No claims found in this arena round.</strong><span>Return to the builder to stage claims.</span></div>`;
      return;
    }

    const advanceStage = nextStage(arenaData.current_stage);
    const terminal = !advanceStage;

    target.innerHTML = groups.map((group) => {
      const runs = orderedRuns(group);
      const ready = claimReadyForDecision(group);
      const focused = runs.some((run) => isFocusedRun(run));
      const failedCount = runs.filter(isRunFailed).length;
      const completeCount = runs.filter((run) => run.completed_stage === arenaData.current_stage).length;
      const agreement = groupAgreement(group);
      const expanded = expandedCompare.has(group.claim_key);
      const pick = preferenceForClaim(group.claim_key);
      const storedSelection = advanceSelections[group.claim_key];
      const defaultSelection = storedSelection
        || (ready && runs.filter((run) => !isRunFailed(run)).length === 1
          ? runs.find((run) => !isRunFailed(run))?.run_id
          : "");

      const columns = [
        { label: "Status", cell: (r) => `<span class="badge ${statusTone(r.current_stage_status || r.status)}">${escapeHtml(String(r.current_stage_status || r.status || "").replace(/_/g, " "))}</span>` },
        {
          label: "Verdict",
          cell: (r) => {
            if (isRunFailed(r)) {
              return `<span class="rating-chip rating-failed">✕ Failed</span>`;
            }
            const rating = runRating(r);
            return rating !== null && Number.isFinite(rating)
              ? ratingChip(rating, { label: r.rating_label !== "Unrated" ? r.rating_label : undefined })
              : `<span class="helper-text">Pending</span>`;
          },
        },
        { label: "Cost", cell: (r) => escapeHtml(formatCurrency(r.usage?.cost_usd || 0)) },
        { label: "Tokens", cell: (r) => escapeHtml(formatTokens(runTokens(r))) },
        { label: "Issues", cell: (r) => String(r.quality_health?.issues_count || 0) },
        ...stageMetricColumns(arenaData.current_stage),
      ];

      if (advanceStage) {
        columns.push({
          label: "Advance",
          cell: (r) => {
            const canAdvance = r.completed_stage === arenaData.current_stage;
            return `
              <label class="checkbox-row">
                <input
                  type="radio"
                  name="${escapeHtml(runSelectionName(group))}"
                  value="${escapeHtml(r.run_id)}"
                  data-advance-claim="${escapeHtml(group.claim_key)}"
                  aria-label="Advance ${escapeHtml(candidateName(r))} to ${escapeHtml(stageLabel(advanceStage))}"
                  ${!ready || !canAdvance ? "disabled" : ""}
                  ${defaultSelection === r.run_id ? "checked" : ""}
                >
                <span>Advance</span>
              </label>
            `;
          },
        });
      } else {
        columns.push({
          label: "Your Pick",
          cell: (r) => {
            if (isRunFailed(r)) {
              return `<span class="helper-text">—</span>`;
            }
            const rating = runRating(r);
            const votable = rating !== null && Number.isFinite(rating);
            return `
              <label class="checkbox-row">
                <input
                  type="radio"
                  name="pref-${escapeHtml(group.claim_key)}"
                  value="${escapeHtml(r.run_id)}"
                  data-pref-claim="${escapeHtml(group.claim_key)}"
                  data-pref-run="${escapeHtml(r.run_id)}"
                  aria-label="Mark ${escapeHtml(candidateName(r))} as the best output for this claim"
                  ${votable ? "" : "disabled"}
                  ${pick?.run_id === r.run_id ? "checked" : ""}
                >
                <span>Best</span>
              </label>
            `;
          },
        });
      }

      columns.push({
        label: "",
        cell: (r) => `
          <div class="inline-actions">
            <a href="/claims/${encodeURIComponent(group.claim_key)}?run_id=${encodeURIComponent(r.run_id)}" class="ghost-button small-button">Detail</a>
            ${isRunFailed(r) ? `<button type="button" class="secondary-button small-button" data-retry-run="${escapeHtml(r.run_id)}">Retry</button>` : ""}
          </div>
        `,
      });

      const items = runs.map((r) => ({ ...r, _candidateColor: r.candidate_color }));
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
        focusTest: (r) => isFocusedRun(r),
      });

      const readyText = ready
        ? (advanceStage ? "Ready — pick a run to advance" : "Comparison ready")
        : "Waiting for candidates";

      return `
        <section class="panel claim-section ${focused ? "focused-claim" : ""}" data-claim-group="${escapeHtml(group.claim_key)}">
          <div class="claim-section-header">
            <div class="stack">
              <h3 class="claim-title">${escapeHtml(group.text)}</h3>
              <div class="record-meta">
                <span>${completeCount}/${runs.length} complete</span>
                ${failedCount ? `<span class="badge error-badge">${failedCount} failed</span>` : ""}
                <span>${escapeHtml(readyText)}</span>
                ${agreementBadge(agreement)}
              </div>
            </div>
            <div class="compare-toggle-row">
              ${terminal && pick ? `<button type="button" class="ghost-button small-button" data-clear-pick="${escapeHtml(group.claim_key)}">Clear pick</button>` : ""}
              <button type="button" class="secondary-button small-button" data-compare-toggle="${escapeHtml(group.claim_key)}" aria-expanded="${expanded ? "true" : "false"}">
                ${expanded ? "Hide Outputs" : "Compare Outputs"}
              </button>
              ${advanceStage ? `
                <label class="checkbox-row">
                  <input type="checkbox" data-skip-claim ${skipSelections[group.claim_key] ? "checked" : ""}>
                  <span>Skip claim</span>
                </label>
              ` : ""}
            </div>
          </div>
          ${tableHtml}
          ${expanded ? renderCompareGrid(group) : ""}
        </section>
      `;
    }).join("");

    target.querySelectorAll("[data-compare-toggle]").forEach((button) => {
      button.addEventListener("click", () => {
        const claimKey = button.dataset.compareToggle;
        if (expandedCompare.has(claimKey)) {
          expandedCompare.delete(claimKey);
        } else {
          expandedCompare.add(claimKey);
        }
        renderClaims();
      });
    });

    target.querySelectorAll("[data-clear-pick]").forEach((button) => {
      button.addEventListener("click", () => {
        savePreference(button.dataset.clearPick, null);
      });
    });

    target.querySelectorAll("[data-retry-run]").forEach((button) => {
      button.addEventListener("click", () => {
        retryRun(button.dataset.retryRun, button);
      });
    });
  }

  // ===== Preferences (best-output picks) =====

  async function savePreference(claimKey, runId) {
    const previous = (arenaData.preferences || {})[claimKey] || null;
    arenaData.preferences = { ...(arenaData.preferences || {}) };
    if (runId) {
      arenaData.preferences[claimKey] = { run_id: runId };
    } else {
      delete arenaData.preferences[claimKey];
    }
    candidateStats = computeCandidateStats();
    renderMatrix();
    renderOverview();
    renderClaims();
    try {
      const data = await fetchJson(`/api/v1/arenas/${encodeURIComponent(config.arenaId)}/preferences`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim_key: claimKey, run_id: runId }),
      });
      arenaData.preferences = data.preferences || {};
    } catch (error) {
      if (previous) {
        arenaData.preferences[claimKey] = previous;
      } else {
        delete arenaData.preferences[claimKey];
      }
      candidateStats = computeCandidateStats();
      renderMatrix();
      renderOverview();
      renderClaims();
      setStatus(byId("workspaceStatus"), {
        title: "Could not save your pick",
        message: error.message,
        tone: "error",
      });
    }
  }

  // ===== Retry failed runs =====

  async function retryRun(runId, button = null) {
    const run = findRunById(runId);
    if (!run) {
      return;
    }
    if (!run.transport_batch_id || !run.transport_claim_id) {
      flashButton(button, { label: "Unavailable", tone: "error" });
      setStatus(byId("workspaceStatus"), {
        title: "Retry unavailable",
        message: `${candidateName(run)} has no transport record to resume from.`,
        tone: "warning",
      });
      return;
    }
    if (!window.confirm(`Re-queue ${candidateName(run)} for this claim? It resumes from its last completed step and may incur additional LLM cost.`)) {
      return;
    }
    const restore = buttonBusy(button, "Queueing…");
    try {
      await fetchJson(
        `/api/v1/claims/${encodeURIComponent(run.transport_batch_id)}/${encodeURIComponent(run.transport_claim_id)}/resume`,
        { method: "POST" }
      );
      restore();
      flashButton(button, { label: "Queued ✓" });
      setStatus(byId("workspaceStatus"), {
        title: "Retry queued",
        message: `${candidateName(run)} was re-queued. Live updates will pick up its progress.`,
        tone: "success",
      });
      refreshWorkspace({ manual: true }).catch(() => {});
    } catch (error) {
      restore();
      flashButton(button, { label: "Retry failed ✗", tone: "error" });
      setStatus(byId("workspaceStatus"), {
        title: "Retry failed",
        message: error.message,
        tone: "error",
      });
    }
  }

  // ===== Export =====

  function csvEscape(value) {
    const text = String(value ?? "");
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function downloadFile(filename, mime, content) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exportRows() {
    const rows = [];
    (arenaData?.claim_groups || []).forEach((group) => {
      orderedRuns(group).forEach((run) => {
        const rating = runRating(run);
        const report = runReport(run);
        rows.push({
          arena_id: arenaData.arena_id || config.arenaId,
          arena_title: arenaData.summary?.title || arenaData.title || "",
          claim_key: group.claim_key,
          claim_text: group.text,
          candidate_prefix: run.candidate_prefix || "",
          candidate_label: run.candidate_label || "",
          provider: run.provider_label || run.provider_id || "",
          model: effectiveModelLabel(run),
          status: run.status || "",
          completed_stage: run.completed_stage || "",
          rating: isScoredVerdict(rating) ? rating : "",
          rating_label: isScoredVerdict(rating) ? ratingLabel(rating) : (run.rating_label || ""),
          your_pick: isPreferredRun(group, run) ? "yes" : "",
          cost_usd: Number(run.usage?.cost_usd || 0),
          total_tokens: runTokens(run),
          elapsed_ms: Number(run.total_elapsed_ms || 0),
          issues: Number(run.quality_health?.issues_count || 0),
          papers_retrieved: (run.claim_data?.processed_papers || []).length,
          queries: (run.claim_data?.semantic_scholar_queries || []).join("; "),
          summary: report?.explanation || "",
          final_reasoning: report?.finalReasoning || "",
        });
      });
    });
    return rows;
  }

  function exportCsv() {
    const rows = exportRows();
    if (!rows.length) {
      setStatus(byId("workspaceStatus"), { title: "Nothing to export", message: "No runs are available yet.", tone: "warning" });
      return;
    }
    const headers = Object.keys(rows[0]);
    const lines = [
      headers.join(","),
      ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
    ];
    downloadFile(`arena_${config.arenaId}_comparison.csv`, "text/csv;charset=utf-8", `${lines.join("\r\n")}\r\n`);
  }

  function exportJson() {
    if (!arenaData) {
      return;
    }
    const payload = {
      arena_id: arenaData.arena_id || config.arenaId,
      title: arenaData.summary?.title || arenaData.title || "",
      current_stage: arenaData.current_stage,
      status: arenaData.summary?.status,
      scorecard: candidateStats.map((entry) => ({
        candidate_prefix: entry.prefix,
        candidate_label: entry.label,
        model: entry.model,
        provider: entry.providerLabel,
        mean_rating: entry.meanRating,
        rated_count: entry.ratedCount,
        verdict_distribution: entry.dist,
        no_evidence_count: entry.noEvidence,
        majority_agreement: entry.agreementPct,
        your_picks: entry.picks,
        failed_runs: entry.failed,
        claims_with_evidence: entry.evidenceClaims,
        papers_total: entry.papersTotal,
        retrieval_overlap: entry.overlapPct,
        total_tokens: entry.tokens,
        actual_cost_usd: entry.cost,
        total_elapsed_ms: entry.elapsedMs,
        issues: entry.issues,
      })),
      preferences: arenaData.preferences || {},
      claims: exportRows(),
    };
    downloadFile(`arena_${config.arenaId}_comparison.json`, "application/json", JSON.stringify(payload, null, 2));
  }

  // ===== History tab =====

  function claimTextForKey(claimKey) {
    return (arenaData.claim_groups || []).find((group) => group.claim_key === claimKey)?.text || claimKey;
  }

  function candidateNameForRunId(runId) {
    const run = findRunById(runId);
    return run ? candidateName(run) : (runId || "Unknown run");
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

  // ===== Continue flow =====

  function collectDecisions() {
    return (arenaData.claim_groups || []).map((group) => {
      const section = sectionForClaimKey(group.claim_key);
      const skipClaim = section
        ? !!section.querySelector("[data-skip-claim]")?.checked
        : !!skipSelections[group.claim_key];
      return {
        claim_key: group.claim_key,
        skip_claim: skipClaim,
        selected_run_id: skipClaim ? null : (selectedRunIdForGroup(group) || null),
      };
    });
  }

  function unresolvedSelectionGroups() {
    return (arenaData?.claim_groups || []).filter((group) => {
      if (!claimReadyForDecision(group)) {
        return false;
      }
      const section = sectionForClaimKey(group.claim_key);
      const skipClaim = !!section?.querySelector("[data-skip-claim]")?.checked;
      if (skipClaim) {
        return false;
      }
      return !selectedRunIdForGroup(group);
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

    const actionLabels = { continue: "Advance", skip: "Skip" };
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
              <span>${escapeHtml(actionLabels[claim.action] || claim.action)}</span>
              ${claim.selected_run_id ? `<span>${escapeHtml(candidateNameForRunId(claim.selected_run_id))}</span>` : ""}
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
    const actionLabels = { continue: "Advance", skip: "Skip" };
    hideStatus(byId("continueModalStatus"));
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
              <span>${escapeHtml(actionLabels[claim.action] || claim.action)}</span>
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

  // ===== Tabs, refresh, init =====

  function activateTab(tabName) {
    document.querySelectorAll("[data-workspace-tab]").forEach((button) => {
      const active = button.dataset.workspaceTab === tabName;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
      button.setAttribute("tabindex", active ? "0" : "-1");
    });
    byId("overviewPane").classList.toggle("hidden", tabName !== "overview");
    byId("claimsPane").classList.toggle("hidden", tabName !== "claims");
    byId("historyPane").classList.toggle("hidden", tabName !== "history");
    byId("overviewPane").toggleAttribute("hidden", tabName !== "overview");
    byId("claimsPane").toggleAttribute("hidden", tabName !== "claims");
    byId("historyPane").toggleAttribute("hidden", tabName !== "history");
  }

  function activeTabName() {
    return document.querySelector("[data-workspace-tab].active")?.dataset.workspaceTab || "overview";
  }

  function shouldAutoRefresh() {
    const status = arenaData?.summary?.status || arenaProgress?.summary?.status;
    // Settled arenas report "completed"/"ready_for_review"/"needs_attention";
    // the resolved check is a safety net for stale summaries (a retry
    // un-resolves a run and flips the arena back to "in_progress").
    return status === "in_progress" && !allRunsResolved();
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
    arenaData.preferences = arenaData.preferences || {};
    arenaProgress = progress;
    candidateStats = computeCandidateStats();

    byId("arenaTitle").textContent = arena.summary?.title || arena.title || arena.arena_id;
    renderSummaryStrip(arena.summary || progress.summary);
    renderStepper();
    renderOverview();
    renderMatrix();
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
        // Persistent context card re-rendered on the 5s auto-refresh — never toast.
        toast: false,
      });
    } else {
      hideStatus(byId("workspaceStatus"));
    }

    if (initial && !hasInitialized) {
      const params = new URLSearchParams(window.location.search);
      const requestedTab = params.get("tab");
      const expandParam = params.get("expand");
      if (expandParam === "all") {
        (arenaData.claim_groups || []).forEach((group) => expandedCompare.add(group.claim_key));
        renderClaims();
      }
      if (requestedTab && ["overview", "claims", "history"].includes(requestedTab)) {
        activateTab(requestedTab);
      } else if (focused) {
        activateTab(focused.status === "processed" ? "claims" : "overview");
      } else {
        const status = arena.summary?.status;
        activateTab(["completed", "ready_for_review", "needs_attention"].includes(status) || allRunsResolved() ? "claims" : "overview");
      }
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
          // Fires on every failed auto-refresh (5s loop) — never toast.
          toast: false,
        });
      }
    } finally {
      refreshInFlight = false;
      updateWorkspaceMeta();
      scheduleAutoRefresh();
    }
  }

  // ===== Event wiring =====

  document.querySelectorAll("[data-workspace-tab]").forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.workspaceTab));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const tabs = Array.from(document.querySelectorAll("[data-workspace-tab]"));
      const currentIndex = tabs.indexOf(button);
      let nextIndex = currentIndex;
      if (event.key === "ArrowRight") nextIndex = (currentIndex + 1) % tabs.length;
      if (event.key === "ArrowLeft") nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = tabs.length - 1;
      tabs[nextIndex].focus();
      activateTab(tabs[nextIndex].dataset.workspaceTab);
    });
  });

  byId("claimGroups").addEventListener("change", (event) => {
    const prefInput = event.target.closest("input[data-pref-run]");
    if (prefInput) {
      savePreference(prefInput.dataset.prefClaim, prefInput.dataset.prefRun);
      return;
    }
    const advanceInput = event.target.closest("input[data-advance-claim]");
    if (advanceInput) {
      advanceSelections[advanceInput.dataset.advanceClaim] = advanceInput.value;
    }
    const skipInput = event.target.closest("input[data-skip-claim]");
    if (skipInput) {
      const section = skipInput.closest("[data-claim-group]");
      if (section) {
        skipSelections[section.dataset.claimGroup] = skipInput.checked;
      }
    }
    continuationPreflight = null;
    renderContinueSummary();
    renderStepper();
  });

  const sortSelect = byId("matrixSortSelect");
  if (sortSelect) {
    sortSelect.addEventListener("change", () => {
      matrixSort = sortSelect.value;
      renderMatrix();
      renderClaims();
    });
  }

  byId("expandAllCompareBtn")?.addEventListener("click", () => {
    (arenaData?.claim_groups || []).forEach((group) => expandedCompare.add(group.claim_key));
    renderClaims();
  });

  byId("collapseAllCompareBtn")?.addEventListener("click", () => {
    expandedCompare.clear();
    renderClaims();
  });

  byId("exportCsvBtn")?.addEventListener("click", exportCsv);
  byId("exportJsonBtn")?.addEventListener("click", exportJson);

  byId("estimateContinueBtn").addEventListener("click", () => {
    const missing = unresolvedSelectionGroups();
    if (missing.length) {
      activateTab("claims");
      // The claims to resolve are in the tab panel above the continue section,
      // so bring the warning (and the panel right below it) into view instead
      // of toasting from the bottom of the page.
      setStatus(byId("workspaceStatus"), {
        title: "Select winners first",
        message: `Pick a winner (or skip) for ${missing.length} claim${missing.length === 1 ? "" : "s"} below before estimating the next stage.`,
        tone: "warning",
        toast: false,
      });
      byId("workspaceStatus").scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    const button = byId("estimateContinueBtn");
    const originalLabel = button.textContent;
    button.disabled = true;
    button.textContent = "Estimating…";
    estimateContinuation()
      .catch((error) => {
        setStatus(byId("workspaceStatus"), {
          title: "Estimate failed",
          message: error.message,
          tone: "error",
        });
      })
      .finally(() => {
        button.disabled = false;
        button.textContent = originalLabel;
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
    const button = byId("confirmContinueBtn");
    const originalLabel = button.textContent;
    button.disabled = true;
    button.textContent = "Continuing…";
    hideStatus(byId("continueModalStatus"));
    continueArena().catch((error) => {
      setStatus(byId("continueModalStatus"), {
        title: "Continuation failed",
        message: error.message,
        tone: "error",
      });
      button.disabled = false;
      button.textContent = originalLabel;
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
