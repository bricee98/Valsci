(() => {
  const { escapeHtml, fetchJson, formatDateTime, hideStatus, setStatus } = window.ValsciUI;
  const byId = (id) => document.getElementById(id);
  let migrationBatches = [];
  let selectedBatchId = null;
  let selectedBatchDetail = null;

  function renderTable() {
    const target = byId("migrationTableBody");
    if (!migrationBatches.length) {
      target.innerHTML = `<tr><td colspan="6"><div class="empty-state"><strong>No legacy batches found.</strong><span>The migration review table will fill in when old transport-only folders are present.</span></div></td></tr>`;
      return;
    }

    target.innerHTML = migrationBatches.map(batch => `
      <tr>
        <td><strong>${escapeHtml(batch.batch_id)}</strong></td>
        <td>${batch.claim_count}</td>
        <td>${escapeHtml(formatDateTime(batch.last_modified_at))}</td>
        <td><span class="badge ${batch.status === "pending" ? "warning-badge" : "neutral-badge"}">${escapeHtml(batch.status.replace(/_/g, " "))}</span></td>
        <td>${escapeHtml((batch.roots || []).join(", "))}</td>
        <td>
          <div class="inline-actions">
            <button type="button" class="secondary-button small-button" data-review="${escapeHtml(batch.batch_id)}">Review contents</button>
            <button type="button" class="primary-button small-button" data-import="${escapeHtml(batch.batch_id)}">Import</button>
          </div>
        </td>
      </tr>
    `).join("");
  }

  function renderReviewActions(detail) {
    const actions = byId("migrationReviewActions");
    actions.classList.toggle("hidden", !detail);

    if (!detail) {
      hideStatus(byId("migrationReviewStatus"));
      return;
    }

    const imported = detail.status === "imported";
    const partiallyImported = detail.status === "partially_imported";
    byId("reviewImportBtn").disabled = imported;
    byId("reviewDeleteBtn").disabled = imported || partiallyImported;

    if (imported) {
      setStatus(byId("migrationReviewStatus"), {
        title: "Legacy copy already imported",
        message: "Deletion stays disabled here for safety once a batch has canonical imported runs.",
        tone: "info",
      });
      return;
    }

    if (partiallyImported) {
      setStatus(byId("migrationReviewStatus"), {
        title: "Partially imported batch",
        message: "Delete is disabled because imported runs already exist. Review, import, or archive from this screen instead.",
        tone: "warning",
      });
      return;
    }

    hideStatus(byId("migrationReviewStatus"));
  }

  function renderReview(detail) {
    const panel = byId("migrationReviewPanel");
    panel.classList.remove("hidden");
    selectedBatchId = detail.batch_id;
    selectedBatchDetail = detail;
    byId("migrationReviewSubtitle").textContent = `${detail.batch_id} / ${detail.claim_count} claim${detail.claim_count === 1 ? "" : "s"} / ${detail.status.replace(/_/g, " ")}`;
    renderReviewActions(detail);
    byId("migrationReviewContent").innerHTML = (detail.claims || []).map(claim => `
      <article class="record-card">
        <div class="panel-header">
          <div>
            <strong>${escapeHtml(claim.text || claim.claim_id)}</strong>
            <p class="panel-subtitle">${escapeHtml(claim.claim_id)} / ${escapeHtml(claim.source_root)}</p>
          </div>
          <span class="badge neutral-badge">${escapeHtml(claim.status)}</span>
        </div>
        <div class="record-meta">
          <span>${escapeHtml(claim.review_type)}</span>
          ${claim.completed_stage ? `<span>${escapeHtml(claim.completed_stage)}</span>` : ""}
          <span>${escapeHtml(formatDateTime(claim.updated_at))}</span>
          <span>${claim.has_report ? "Has report" : "No report yet"}</span>
        </div>
      </article>
    `).join("") || `<div class="empty-state"><strong>No claim preview available.</strong></div>`;
  }

  async function loadBatches() {
    const data = await fetchJson("/api/v1/migration/batches");
    migrationBatches = data.batches || [];
    renderTable();
  }

  async function reviewBatch(batchId) {
    const detail = await fetchJson(`/api/v1/migration/batches/${encodeURIComponent(batchId)}`);
    renderReview(detail);
  }

  async function importBatch(batchId, archiveAfter = false) {
    await fetchJson(`/api/v1/migration/batches/${encodeURIComponent(batchId)}/import`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ archive_after: archiveAfter }),
    });
    setStatus(byId("migrationStatus"), {
      title: archiveAfter ? "Batch imported and archived" : "Batch imported",
      message: `${batchId} has been imported into the claim store.`,
      tone: "success",
    });
    await loadBatches();
    await reviewBatch(batchId).catch(() => {});
  }

  async function deleteBatch(batchId) {
    await fetchJson(`/api/v1/migration/batches/${encodeURIComponent(batchId)}`, {
      method: "DELETE",
    });
    setStatus(byId("migrationStatus"), {
      title: "Legacy batch deleted",
      message: `${batchId} was removed from the legacy transport folders.`,
      tone: "warning",
    });
    byId("migrationReviewPanel").classList.add("hidden");
    byId("migrationReviewActions").classList.add("hidden");
    hideStatus(byId("migrationReviewStatus"));
    selectedBatchId = null;
    selectedBatchDetail = null;
    await loadBatches();
  }

  async function importAll() {
    await fetchJson("/api/v1/migration/import_all", { method: "POST" });
    setStatus(byId("migrationStatus"), {
      title: "Pending legacy batches imported",
      message: "All pending legacy folders were imported into the claim store.",
      tone: "success",
    });
    await loadBatches();
  }

  byId("refreshMigrationBtn").addEventListener("click", () => loadBatches().catch(error => setStatus(byId("migrationStatus"), { title: "Refresh failed", message: error.message, tone: "error" })));
  byId("importAllBtn").addEventListener("click", () => importAll().catch(error => setStatus(byId("migrationStatus"), { title: "Import failed", message: error.message, tone: "error" })));
  byId("migrationTableBody").addEventListener("click", event => {
    const reviewButton = event.target.closest("[data-review]");
    if (reviewButton) {
      reviewBatch(reviewButton.dataset.review).catch(error => setStatus(byId("migrationStatus"), { title: "Review failed", message: error.message, tone: "error" }));
      return;
    }
    const importButton = event.target.closest("[data-import]");
    if (importButton) {
      importBatch(importButton.dataset.import, true).catch(error => setStatus(byId("migrationStatus"), { title: "Import failed", message: error.message, tone: "error" }));
    }
  });

  byId("reviewImportBtn").addEventListener("click", () => {
    if (!selectedBatchId) {
      return;
    }
    importBatch(selectedBatchId, true).catch(error => setStatus(byId("migrationStatus"), { title: "Import failed", message: error.message, tone: "error" }));
  });

  byId("reviewDeleteBtn").addEventListener("click", () => {
    if (!selectedBatchId || !selectedBatchDetail) {
      return;
    }
    if (window.confirm(`Delete legacy batch ${selectedBatchId}? This only removes the legacy copy.`)) {
      deleteBatch(selectedBatchId).catch(error => setStatus(byId("migrationStatus"), { title: "Delete failed", message: error.message, tone: "error" }));
    }
  });

  hideStatus(byId("migrationStatus"));
  hideStatus(byId("migrationReviewStatus"));
  loadBatches().catch(error => {
    setStatus(byId("migrationStatus"), { title: "Migration review failed to load", message: error.message, tone: "error" });
  });
})();
