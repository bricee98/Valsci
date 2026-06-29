(() => {
  const {
    escapeHtml,
    fetchJson,
    formatDateTime,
    hideStatus,
    setStatus,
    revealPanel,
    flashButton,
    buttonBusy,
  } = window.ValsciUI;

  const byId = (id) => document.getElementById(id);

  const statusBadgeClass = {
    queued: "warning-badge",
    running: "warning-badge",
    cancel_requested: "warning-badge",
    success: "success-badge",
    failed: "error-badge",
    cancelled: "neutral-badge",
    interrupted: "error-badge",
  };

  const indexBadgeClass = {
    ready: "success-badge",
    needs_attention: "warning-badge",
    missing: "warning-badge",
    not_applicable: "neutral-badge",
  };

  let currentState = null;
  let selectedReleaseId = null;
  let activeJobId = null;
  let pollTimer = null;
  // Dataset checkbox selection for the currently selected release. Preserved
  // across polling/rerenders; reset (re-defaulted) only when the release changes.
  let releaseDatasetSelection = null; // { releaseId, names: Set<string> }
  // The Current Job panel shows the active job, or — when idle — the job the
  // user pinned via "View log", or the most recent job. This keeps a finished
  // job's log (e.g. a failure) reachable instead of vanishing on completion.
  let selectedJobId = null; // job pinned to the Current Job panel by the user
  let displayedJobId = null; // job currently rendered in the Current Job panel
  let currentRecentJobs = []; // last-seen recent jobs list (for re-highlighting)

  function setApiKeyMissingStatus() {
    const target = byId("dataStatusCard");
    if (!target) return;
    target.className = "status-card warning-card";
    target.innerHTML = `
      <strong>Semantic Scholar API key is missing</strong>
      <span>
        Downloads and verification need SEMANTIC_SCHOLAR_API_KEY.
        <a href="/settings#SEMANTIC_SCHOLAR_API_KEY">Add it in Settings</a>.
        Existing local files can still be inspected and indexed.
      </span>
    `;
    target.classList.remove("hidden");
  }

  function statusLabel(value) {
    return String(value || "idle").replace(/_/g, " ");
  }

  function setBadge(target, status) {
    if (!target) return;
    const normalized = status || "idle";
    target.className = `badge ${statusBadgeClass[normalized] || "neutral-badge"}`;
    target.textContent = statusLabel(normalized);
  }

  function selectedRelease() {
    return (currentState?.releases || []).find((release) => release.release_id === selectedReleaseId) || null;
  }

  function activeJobRunning() {
    return Boolean(activeJobId);
  }

  function renderSummary(state) {
    const active = selectedRelease() || state.active_release;
    const manifest = state.mini_manifest || {};
    byId("dataSummary").innerHTML = `
      <div class="summary-cell">
        <span class="label">Latest Release</span>
        <span class="value">${escapeHtml(state.latest_release || "None")}</span>
      </div>
      <div class="summary-cell">
        <span class="label">Local Releases</span>
        <span class="value">${(state.releases || []).length}</span>
      </div>
      <div class="summary-cell">
        <span class="label">Selected Size</span>
        <span class="value">${escapeHtml(active?.size_label || "0 B")}</span>
      </div>
      <div class="summary-cell">
        <span class="label">S2 API Key</span>
        <span class="value">${state.api_key_present ? "Configured" : "Missing"}</span>
      </div>
    `;

    if (!state.api_key_present) {
      setApiKeyMissingStatus();
    } else if (!state.latest_release) {
      setStatus(byId("dataStatusCard"), {
        title: "No local Semantic Scholar release found",
        message: manifest.exists
          ? "Create a curated mini release or download a full release to prepare local evidence lookup."
          : "Create a release from the Mendelian mini manifest or download selected full datasets.",
        tone: "info",
        // Persistent inline card re-rendered on the 3s poll loop — never toast.
        toast: false,
      });
    } else {
      hideStatus(byId("dataStatusCard"));
    }
  }

  function renderReleaseSelector(state) {
    const releases = state.releases || [];
    if (!selectedReleaseId || !releases.some((release) => release.release_id === selectedReleaseId)) {
      selectedReleaseId = state.latest_release || (releases[0]?.release_id || "");
    }
    if (!releases.length) {
      byId("releaseSelect").innerHTML = `<option value="">No local releases</option>`;
      byId("releaseSelect").disabled = true;
      return;
    }
    byId("releaseSelect").disabled = false;
    byId("releaseSelect").innerHTML = releases.map((release) => `
      <option value="${escapeHtml(release.release_id)}" ${release.release_id === selectedReleaseId ? "selected" : ""}>
        ${escapeHtml(release.release_id)}${release.is_mini ? " (mini)" : ""}
      </option>
    `).join("");
  }

  function indexSummaryText(status) {
    if (!status || status.state === "not_applicable") return "No index mappings";
    if (status.state === "ready") return `${Number(status.present || 0).toLocaleString()} ready`;
    if (status.state === "missing") return `${Number(status.missing || 0).toLocaleString()} missing`;
    return `${Number(status.missing || 0).toLocaleString()} missing, ${Number(status.unhealthy || 0).toLocaleString()} unhealthy`;
  }

  function recordsWrittenPills(release) {
    const entries = Object.entries(release.records_written || {});
    if (!release.is_mini || !entries.length) return "";
    return `
      <div class="pill-row">
        ${entries.map(([dataset, count]) => `
          <span class="pill">${escapeHtml(dataset)}: ${Number(count || 0).toLocaleString()}</span>
        `).join("")}
      </div>
    `;
  }

  function manifestCoverageCard(release) {
    const coverage = release.manifest_coverage;
    if (!coverage) return "";
    const missingEntries = Object.entries(coverage.missing || {});
    if (!missingEntries.length) {
      return `
        <div class="status-card success-card">
          <strong>Mini release matches the current manifest.</strong>
          <span>Requested records are present locally for the tracked Mendelian mini corpus.</span>
        </div>
      `;
    }
    const missingText = missingEntries
      .map(([dataset, count]) => `${dataset}: ${Number(count || 0).toLocaleString()}`)
      .join(", ");
    return `
      <div class="status-card warning-card">
        <strong>Mini release rows need rebuild.</strong>
        <span>The indices can be healthy while the compact mini data is stale. The current manifest requests rows that this local release does not contain (${escapeHtml(missingText)}). Rebuild the mini release to fetch those rows.</span>
        <div class="inline-actions">
          <button type="button" class="primary-button small-button" id="rebuildMiniReleaseBtn">Rebuild Mini Release</button>
        </div>
      </div>
    `;
  }

  function releaseMaintenanceSubtitle(release, indexStatus) {
    const base = indexSummaryText(indexStatus);
    if (release.manifest_coverage?.state === "stale") {
      return `${base}; mini rows stale`;
    }
    return base;
  }

  function pickerDatasets(release) {
    return (release.datasets || []).filter((dataset) => dataset.indices?.length || dataset.exists);
  }

  function ensureReleaseSelection(release) {
    if (!releaseDatasetSelection || releaseDatasetSelection.releaseId !== release.release_id) {
      const defaults = pickerDatasets(release)
        .filter((dataset) => dataset.exists)
        .map((dataset) => dataset.name);
      releaseDatasetSelection = { releaseId: release.release_id, names: new Set(defaults) };
    }
    return releaseDatasetSelection;
  }

  function releaseDatasetPicker(release) {
    const datasets = pickerDatasets(release);
    if (!datasets.length) {
      return `<div class="empty-state"><strong>No datasets found for this release.</strong></div>`;
    }
    const selection = ensureReleaseSelection(release);
    return `
      <div class="release-dataset-picker" id="releaseDatasetPicker">
        ${datasets.map((dataset) => `
          <label class="dataset-option">
            <input type="checkbox" value="${escapeHtml(dataset.name)}" ${selection.names.has(dataset.name) ? "checked" : ""}>
            <span>
              <strong>${escapeHtml(dataset.label || dataset.name)}</strong>
              <small>${escapeHtml(dataset.size_label || "0 B")} · ${dataset.fully_indexed ? "indexed" : "index needed"}</small>
            </span>
          </label>
        `).join("")}
      </div>
    `;
  }

  function renderReleaseDetails() {
    const release = selectedRelease();
    if (!release) {
      byId("releaseDetails").innerHTML = `
        <div class="empty-state">
          <strong>No local release selected.</strong>
          <span>Create a mini release from the Mendelian manifest or download a full Semantic Scholar release.</span>
        </div>
      `;
      return;
    }
    const indexStatus = release.index_status || {};
    byId("releaseDetails").innerHTML = `
      <div class="summary-strip">
        <div class="summary-cell"><span class="label">Type</span><span class="value">${release.is_mini ? "Mini" : "Full"}</span></div>
        <div class="summary-cell"><span class="label">Size</span><span class="value">${escapeHtml(release.size_label)}</span></div>
        <div class="summary-cell"><span class="label">Topic</span><span class="value">${escapeHtml(release.topic_label || "General")}</span></div>
        <div class="summary-cell">
          <span class="label">Index</span>
          <span class="value"><span class="badge ${indexBadgeClass[indexStatus.state] || "neutral-badge"}">${escapeHtml(indexStatus.label || "Unknown")}</span></span>
        </div>
      </div>
      ${recordsWrittenPills(release)}
      ${manifestCoverageCard(release)}
      <section class="release-maintenance">
        <div class="panel-header compact-header">
          <div>
            <h3 class="panel-title">Release Maintenance</h3>
            <p class="panel-subtitle">${escapeHtml(releaseMaintenanceSubtitle(release, indexStatus))}</p>
          </div>
          <div class="inline-actions">
            <button type="button" class="secondary-button small-button" id="verifyReleaseBtn">Verify</button>
            <button type="button" class="primary-button small-button" id="indexReleaseBtn">Re-index</button>
          </div>
        </div>
        ${releaseDatasetPicker(release)}
      </section>
    `;
    syncActionButtons();
  }

  function indexSummary(dataset) {
    if (!dataset.indices.length) {
      return `<span class="badge neutral-badge">No index mapping</span>`;
    }
    return dataset.indices.map((index) => {
      if (!index.exists) {
        return `<span class="badge warning-badge">${escapeHtml(index.id_type)} missing</span>`;
      }
      const cls = index.healthy ? "success-badge" : "error-badge";
      return `<span class="badge ${cls}">${escapeHtml(index.id_type)} ${Number(index.entry_count || 0).toLocaleString()}</span>`;
    }).join(" ");
  }

  function renderDatasetTable() {
    const release = selectedRelease();
    if (!release) {
      byId("datasetTable").innerHTML = `<div class="empty-state"><strong>No dataset rows yet.</strong><span>Create or select a release to inspect coverage.</span></div>`;
      return;
    }
    byId("datasetTable").innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Local Files</th>
            <th>Size</th>
            <th>Index</th>
            <th>Role</th>
          </tr>
        </thead>
        <tbody>
          ${(release.datasets || []).map((dataset) => `
            <tr>
              <td>
                <strong>${escapeHtml(dataset.label || dataset.name)}</strong>
                <span class="data-path">${escapeHtml(dataset.name)}</span>
              </td>
              <td>
                ${dataset.exists && dataset.file_count
                  ? `<span class="badge success-badge">${dataset.file_count.toLocaleString()} file${dataset.file_count === 1 ? "" : "s"}</span>`
                  : `<span class="badge warning-badge">Missing</span>`}
              </td>
              <td>${escapeHtml(dataset.size_label || "0 B")}</td>
              <td><div class="pill-row">${indexSummary(dataset)}</div></td>
              <td>${escapeHtml(dataset.note || "")}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  }

  function renderManifest(state) {
    const manifest = state.mini_manifest || {};
    // The manifest is selected by filename only (resolved from semantic_scholar/manifests/).
    byId("miniManifestPath").value = manifest.manifest || "";
    const counts = manifest.dataset_id_counts || {};
    const countText = Object.keys(counts).length
      ? Object.entries(counts).map(([dataset, count]) => `${dataset}: ${Number(count || 0).toLocaleString()}`).join(", ")
      : `${manifest.corpus_id_count || 0} corpus IDs, ${manifest.author_id_count || 0} author IDs`;
    let message;
    let tone;
    if (manifest.error) {
      message = `Manifest problem: ${manifest.error}`;
      tone = "warning";
    } else if (manifest.exists) {
      message = `Manifest ${manifest.manifest || ""}: ${countText}. Valsci will fetch matching rows from Semantic Scholar dataset shards.`;
      tone = "info";
    } else {
      message = `The selected manifest (${manifest.manifest || "unknown"}) is missing from semantic_scholar/manifests/.`;
      tone = "warning";
    }
    setStatus(byId("miniManifestSummary"), {
      title: "Curated Mendelian mini",
      message,
      tone,
      // Persistent inline summary re-rendered on the 3s poll loop — never toast,
      // or an off-screen card spawns a new toast on every refresh.
      toast: false,
    });
  }

  function renderWizardDatasetPicker(state) {
    const options = state.dataset_options || [];
    byId("wizardDatasetPicker").innerHTML = options.map((dataset) => `
      <label class="dataset-option">
        <input type="checkbox" value="${escapeHtml(dataset.name)}" ${dataset.default ? "checked" : ""}>
        <span>
          <strong>${escapeHtml(dataset.label)}</strong>
          <small>${escapeHtml(dataset.note || "")}</small>
        </span>
      </label>
    `).join("");
  }

  function selectedWizardDatasets() {
    return Array.from(byId("wizardDatasetPicker").querySelectorAll("input[type='checkbox']:checked"))
      .map((input) => input.value);
  }

  function selectedReleaseDatasets() {
    return Array.from(document.querySelectorAll("#releaseDatasetPicker input[type='checkbox']:checked"))
      .map((input) => input.value);
  }

  function selectedNewReleaseType() {
    return document.querySelector("input[name='newReleaseType']:checked")?.value || "mini";
  }

  function syncWizardUi() {
    const type = selectedNewReleaseType();
    byId("miniReleaseFields").classList.toggle("hidden", type !== "mini");
    byId("fullReleaseFields").classList.toggle("hidden", type !== "full");
  }

  function openNewReleaseWizard() {
    hideStatus(byId("newReleaseStatus"));
    byId("newReleaseModal").classList.remove("hidden");
    syncWizardUi();
  }

  function closeNewReleaseWizard() {
    byId("newReleaseModal").classList.add("hidden");
  }

  function syncActionButtons() {
    const running = activeJobRunning();
    byId("newReleaseBtn").disabled = running;
    const verifyButton = byId("verifyReleaseBtn");
    const indexButton = byId("indexReleaseBtn");
    const rebuildButton = byId("rebuildMiniReleaseBtn");
    if (verifyButton) verifyButton.disabled = running || !selectedRelease();
    if (indexButton) indexButton.disabled = running || !selectedRelease();
    if (rebuildButton) rebuildButton.disabled = running;
    byId("startNewReleaseBtn").disabled = running;
  }

  function renderJob(job) {
    const isActive = Boolean(job && ["queued", "running", "cancel_requested"].includes(job.status));
    activeJobId = isActive ? job.job_id : null;
    displayedJobId = job ? job.job_id : null;
    setBadge(byId("jobStatusBadge"), job?.status || "idle");
    byId("cancelDataJobBtn").classList.toggle("hidden", !job || !["queued", "running", "cancel_requested"].includes(job.status));
    if (!job) {
      byId("jobSummary").innerHTML = `<div class="empty-state"><strong>No data job running.</strong><span>Release jobs will appear here.</span></div>`;
      byId("jobLog").textContent = "";
      syncActionButtons();
      return;
    }
    const exitCodeCell = (job.exit_code === null || job.exit_code === undefined)
      ? ""
      : `<div class="summary-cell"><span class="label">Exit Code</span><span class="value">${escapeHtml(String(job.exit_code))}</span></div>`;
    const stderrTail = Array.isArray(job.stderr_tail) ? job.stderr_tail : [];
    const stderrBlock = stderrTail.length
      ? `<div class="status-card warning-card"><strong>stderr (last ${stderrTail.length})</strong><pre class="job-log stderr-log">${escapeHtml(stderrTail.join("\n"))}</pre></div>`
      : "";
    byId("jobSummary").innerHTML = `
      <div class="summary-strip">
        <div class="summary-cell"><span class="label">Operation</span><span class="value">${escapeHtml(job.operation)}</span></div>
        <div class="summary-cell"><span class="label">Started</span><span class="value">${escapeHtml(formatDateTime(job.started_at || job.created_at))}</span></div>
        ${exitCodeCell}
      </div>
      <span class="data-path">${escapeHtml(job.command_display || "")}</span>
      ${stderrBlock}
    `;
    const logText = (job.logs || [])
      .map((entry) => (entry.stream === "stderr" ? `[stderr] ${entry.line}` : entry.line))
      .join("\n");
    byId("jobLog").textContent = logText;
    byId("jobLog").scrollTop = byId("jobLog").scrollHeight;
    syncActionButtons();
  }

  function renderRecentJobs(jobs) {
    const recent = jobs || [];
    currentRecentJobs = recent;
    if (!recent.length) {
      byId("recentDataJobs").innerHTML = `<div class="empty-state"><strong>No data jobs yet.</strong><span>Downloader runs will appear here.</span></div>`;
      return;
    }
    byId("recentDataJobs").innerHTML = recent.map((job) => `
      <article class="record-card ${job.job_id === displayedJobId ? "record-card-active" : ""}">
        <div class="record-meta">
          <span class="badge ${statusBadgeClass[job.status] || "neutral-badge"}">${escapeHtml(statusLabel(job.status))}</span>
          <span>${escapeHtml(job.operation || "operation")}</span>
          <span>${escapeHtml(formatDateTime(job.created_at))}</span>
          ${(job.exit_code === null || job.exit_code === undefined) ? "" : `<span>exit ${escapeHtml(String(job.exit_code))}</span>`}
        </div>
        <span class="data-path">${escapeHtml(job.command_display || "")}</span>
        ${job.status === "failed" && job.error ? `<span class="data-path error-text">${escapeHtml(job.error)}</span>` : ""}
        <div class="inline-actions">
          <button type="button" class="secondary-button small-button" data-view-log="${escapeHtml(job.job_id)}">
            ${job.job_id === displayedJobId ? "Viewing log" : "View log"}
          </button>
        </div>
      </article>
    `).join("");
  }

  function jobToDisplay(data) {
    // Prefer a live job; otherwise the user's pinned job, otherwise the latest.
    if (data.active_job) return data.active_job;
    const jobs = data.jobs || [];
    if (selectedJobId) {
      const pinned = jobs.find((job) => job.job_id === selectedJobId);
      if (pinned) return pinned;
    }
    return jobs[0] || null;
  }

  function renderAll(data) {
    currentState = data.state || {};
    renderReleaseSelector(currentState);
    renderSummary(currentState);
    renderReleaseDetails();
    renderDatasetTable();
    renderManifest(currentState);
    renderWizardDatasetPicker(currentState);
    renderJob(jobToDisplay(data));
    renderRecentJobs(data.jobs || []);
    byId("releaseInput").placeholder = currentState.latest_release || "latest";
  }

  async function refreshData() {
    const data = await fetchJson("/api/v1/data/status");
    renderAll(data);
    return data;
  }

  async function poll() {
    try {
      const previousActiveJobId = activeJobId;
      const data = await refreshData();
      const job = data.active_job;
      if (!job && previousActiveJobId) {
        const completed = await fetchJson(`/api/v1/data/jobs/${previousActiveJobId}`).catch(() => null);
        if (completed?.job) {
          renderJob(completed.job);
          handleCompletedJob(completed.job, data.state || {});
        }
        activeJobId = null;
      }
    } catch (error) {
      setStatus(byId("dataStatusCard"), {
        title: "Could not refresh data state",
        message: error.message,
        tone: "error",
        // Re-rendered on the 3s poll loop — a persistent fetch failure must not
        // spawn a toast on every retry.
        toast: false,
      });
    }
  }

  function jobLogText(job) {
    return (job.logs || []).map((entry) => entry.line || "").join("\n");
  }

  function handleCompletedJob(job, state) {
    if (!job || !["success", "failed", "cancelled", "interrupted"].includes(job.status)) return;
    const release = selectedRelease();
    const logText = jobLogText(job);
    const staleMini = release?.manifest_coverage?.state === "stale";

    if (job.operation === "index" && job.status === "success" && staleMini) {
      setStatus(byId("releaseActionStatus"), {
        title: "Indices rebuilt",
        message: "Reindex refreshed lookup indices, but the mini release is still missing manifest rows. Rebuild the mini release to clear the stale-data warning.",
        tone: "warning",
      });
      return;
    }
    if (job.operation === "verify" && job.status === "failed" && /Mini corpus release is stale/i.test(logText)) {
      setStatus(byId("releaseActionStatus"), {
        title: "Verification found stale mini rows",
        message: "Verify is working: it failed because the local mini release is missing rows requested by the current manifest. Rebuild the mini release, then verify again.",
        tone: "warning",
      });
      return;
    }
    if (job.operation === "mini" && job.status === "success") {
      setStatus(byId("releaseActionStatus"), {
        title: "Mini release rebuilt",
        message: "The curated mini release finished rebuilding. Refresh or select it to inspect the updated dataset and index counts.",
        tone: "success",
      });
      return;
    }
    if (job.status === "success") {
      setStatus(byId("releaseActionStatus"), {
        title: `${statusLabel(job.operation)} complete`,
        message: "The data job completed successfully.",
        tone: "success",
      });
      return;
    }
    if (job.status === "failed") {
      setStatus(byId("releaseActionStatus"), {
        title: `${statusLabel(job.operation)} failed`,
        message: "The full output is shown in the Current Job panel. You can also reopen it anytime with “View log” under Recent Data Jobs.",
        tone: "error",
      });
    }
  }

  async function startJob(payload, statusTarget) {
    const data = await fetchJson("/api/v1/data/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    activeJobId = data.job.job_id;
    selectedJobId = null; // follow the new active job in the Current Job panel
    renderJob(data.job);
    hideStatus(statusTarget);
    // The Current Job panel can sit below the fold (single-column layouts), so
    // bring it into view whenever a job starts streaming output.
    revealPanel(byId("jobLog").closest(".panel"));
    poll();
    return data.job;
  }

  async function startNewReleaseJob() {
    const type = selectedNewReleaseType();
    const payload = {
      operation: type,
      release: byId("releaseInput").value.trim() || "latest",
      datasets: selectedWizardDatasets(),
    };
    if (type === "mini") {
      payload.manifest = byId("miniManifestPath").value.trim();
    }
    if (type === "full") {
      if (!payload.datasets.length) {
        setStatus(byId("newReleaseStatus"), {
          title: "Choose at least one dataset",
          message: "Select datasets before starting a full release download.",
          tone: "warning",
        });
        return;
      }
      if (!window.confirm("Full Semantic Scholar downloads can consume a very large amount of disk space. Start this download now?")) {
        return;
      }
    }
    await startJob(payload, byId("newReleaseStatus"));
    closeNewReleaseWizard();
  }

  async function rebuildMiniRelease() {
    const manifestName = currentState?.mini_manifest?.manifest || "";
    if (currentState?.mini_manifest?.error) {
      setStatus(byId("releaseActionStatus"), {
        title: "Mini manifest unavailable",
        message: currentState.mini_manifest.error,
        tone: "warning",
      });
      return;
    }
    await startJob({ operation: "mini", manifest: manifestName }, byId("releaseActionStatus"));
  }

  async function startSelectedReleaseJob(operation) {
    const release = selectedRelease();
    if (!release) return;
    const datasets = selectedReleaseDatasets();
    if (!datasets.length && !release.is_mini) {
      setStatus(byId("releaseActionStatus"), {
        title: "Choose at least one dataset",
        message: "Select datasets before running this release action.",
        tone: "warning",
      });
      return;
    }
    const payload = {
      operation,
      release: release.release_id,
      datasets,
      mini: operation === "verify" && release.is_mini,
    };
    if (payload.mini) {
      payload.manifest = currentState?.mini_manifest?.manifest || "";
    }
    await startJob(payload, byId("releaseActionStatus"));
  }

  async function cancelJob() {
    if (!activeJobId) return;
    const data = await fetchJson(`/api/v1/data/jobs/${activeJobId}/cancel`, { method: "POST" });
    renderJob(data.job);
  }

  async function viewJobLog(jobId) {
    // Pin the job so polling keeps it in the Current Job panel, then load it.
    selectedJobId = jobId;
    const data = await fetchJson(`/api/v1/data/jobs/${jobId}`).catch(() => null);
    if (data?.job) {
      renderJob(data.job);
      renderRecentJobs(currentRecentJobs);
      byId("jobLog").scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function bindEvents() {
    byId("refreshDataBtn").addEventListener("click", () => {
      const button = byId("refreshDataBtn");
      const restore = buttonBusy(button, "Refreshing…");
      Promise.resolve(refreshData()).then(() => {
        restore();
        flashButton(button, { label: "Refreshed ✓", duration: 1200 });
      }).catch(() => {
        restore();
        flashButton(button, { label: "Failed ✗", tone: "error" });
      });
    });
    byId("releaseSelect").addEventListener("change", (event) => {
      selectedReleaseId = event.target.value;
      // Switching releases resets dataset checkbox selections to defaults.
      releaseDatasetSelection = null;
      renderSummary(currentState || {});
      renderReleaseDetails();
      renderDatasetTable();
    });
    byId("newReleaseBtn").addEventListener("click", openNewReleaseWizard);
    byId("closeNewReleaseBtn").addEventListener("click", closeNewReleaseWizard);
    byId("cancelNewReleaseBtn").addEventListener("click", closeNewReleaseWizard);
    document.querySelectorAll("input[name='newReleaseType']").forEach((input) => {
      input.addEventListener("change", syncWizardUi);
    });
    byId("startNewReleaseBtn").addEventListener("click", () => {
      const restore = buttonBusy(byId("startNewReleaseBtn"), "Starting…");
      startNewReleaseJob().then(restore).catch((error) => {
        restore();
        setStatus(byId("newReleaseStatus"), {
          title: "Could not start release job",
          message: error.message,
          tone: "error",
        });
      });
    });
    byId("releaseDetails").addEventListener("click", (event) => {
      const runReleaseAction = (button, busyLabel, action, errorTitle) => {
        const restore = buttonBusy(button, busyLabel);
        action().then(restore).catch((error) => {
          restore();
          flashButton(button, { label: "Failed ✗", tone: "error" });
          setStatus(byId("releaseActionStatus"), {
            title: errorTitle,
            message: error.message,
            tone: "error",
          });
        });
      };
      const rebuildBtn = event.target.closest("#rebuildMiniReleaseBtn");
      if (rebuildBtn) {
        runReleaseAction(rebuildBtn, "Starting…", rebuildMiniRelease, "Could not rebuild mini release");
      }
      const verifyBtn = event.target.closest("#verifyReleaseBtn");
      if (verifyBtn) {
        runReleaseAction(verifyBtn, "Starting…", () => startSelectedReleaseJob("verify"), "Verify failed");
      }
      const indexBtn = event.target.closest("#indexReleaseBtn");
      if (indexBtn) {
        const confirmed = window.confirm(
          "Re-index rebuilds only the lookup indices for this release. " +
          "Dataset files are not changed, re-downloaded, or deleted — only the binary indices are rebuilt. Continue?"
        );
        if (!confirmed) return;
        runReleaseAction(indexBtn, "Starting…", () => startSelectedReleaseJob("index"), "Re-index failed");
      }
    });
    byId("releaseDetails").addEventListener("change", (event) => {
      const input = event.target;
      if (!input || !input.matches("#releaseDatasetPicker input[type='checkbox']")) return;
      const release = selectedRelease();
      if (!release) return;
      const selection = ensureReleaseSelection(release);
      if (input.checked) {
        selection.names.add(input.value);
      } else {
        selection.names.delete(input.value);
      }
    });
    byId("cancelDataJobBtn").addEventListener("click", () => {
      cancelJob().catch((error) => setStatus(byId("dataStatusCard"), {
        title: "Could not cancel data job",
        message: error.message,
        tone: "error",
      }));
    });
    byId("recentDataJobs").addEventListener("click", (event) => {
      const button = event.target.closest("[data-view-log]");
      if (!button) return;
      viewJobLog(button.dataset.viewLog).catch((error) => setStatus(byId("dataStatusCard"), {
        title: "Could not load job log",
        message: error.message,
        tone: "error",
      }));
    });
  }

  // ---- Remote content-fetch toggle (FETCH_REMOTE_CONTENT_ON_MISS) ----------

  function setRemoteFetchStatus(message, tone) {
    const el = byId("remoteFetchStatus");
    if (!el) return;
    el.textContent = message || "";
    el.style.color = tone === "error" ? "var(--accent)" : "var(--ink-muted)";
  }

  async function loadRemoteFetchToggle() {
    const toggle = byId("remoteFetchToggle");
    if (!toggle) return;
    try {
      const state = await fetchJson("/api/v1/settings/env");
      const entry = (state.entries || []).find((e) => e.env_key === "FETCH_REMOTE_CONTENT_ON_MISS");
      toggle.checked = Boolean(entry && entry.value === true);
      setRemoteFetchStatus(toggle.checked ? "On — missing papers are fetched from the web." : "Off — only local corpus content is used.");
    } catch (error) {
      setRemoteFetchStatus("Could not load the setting.", "error");
    }
  }

  function bindRemoteFetchToggle() {
    const toggle = byId("remoteFetchToggle");
    if (!toggle) return;
    toggle.addEventListener("change", async () => {
      const desired = toggle.checked;
      toggle.disabled = true;
      setRemoteFetchStatus("Saving…");
      try {
        await fetchJson("/api/v1/settings/env", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ updates: { FETCH_REMOTE_CONTENT_ON_MISS: desired } }),
        });
        setRemoteFetchStatus(desired
          ? "On — saved. The processor applies this within a few seconds."
          : "Off — saved. Only local corpus content is used.");
      } catch (error) {
        toggle.checked = !desired;
        setRemoteFetchStatus(`Save failed: ${error.message}`, "error");
      } finally {
        toggle.disabled = false;
      }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    bindRemoteFetchToggle();
    syncWizardUi();
    refreshData().catch((error) => {
      setStatus(byId("dataStatusCard"), {
        title: "Could not load data state",
        message: error.message,
        tone: "error",
      });
    });
    loadRemoteFetchToggle();
    pollTimer = window.setInterval(poll, 3000);
    window.addEventListener("beforeunload", () => window.clearInterval(pollTimer));
  });
})();
