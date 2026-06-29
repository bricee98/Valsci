(() => {
  const { escapeHtml, fetchJson, formatDateTime, hideStatus, setStatus, revealPanel, flashButton, buttonBusy } = window.ValsciUI;
  const byId = (id) => document.getElementById(id);
  let migrationBatches = [];
  let selectedBatchId = null;
  let selectedBatchDetail = null;

  function clearReview() {
    byId("migrationReviewPanel").classList.add("hidden");
    byId("migrationReviewActions").classList.add("hidden");
    hideStatus(byId("migrationReviewStatus"));
    selectedBatchId = null;
    selectedBatchDetail = null;
  }

  function renderTable() {
    const target = byId("migrationTableBody");
    if (!migrationBatches.length) {
      target.innerHTML = `<tr><td colspan="6"><div class="empty-state"><strong>No legacy batches found.</strong><span>The migration review table will fill in when old transport-only folders are present.</span></div></td></tr>`;
      return;
    }

    target.innerHTML = migrationBatches.map(batch => {
      const imported = batch.status === "imported";
      const partiallyImported = batch.status === "partially_imported";
      const actionLabel = imported ? "Archive" : partiallyImported ? "Import Remaining" : "Import";
      const badgeClass = batch.status === "pending" || partiallyImported ? "warning-badge" : "neutral-badge";
      return `
        <tr>
          <td><strong>${escapeHtml(batch.batch_id)}</strong></td>
          <td>${batch.claim_count}</td>
          <td>${escapeHtml(formatDateTime(batch.last_modified_at))}</td>
          <td><span class="badge ${badgeClass}">${escapeHtml(batch.status.replace(/_/g, " "))}</span></td>
          <td>${escapeHtml((batch.roots || []).join(", "))}</td>
          <td>
            <div class="inline-actions">
              <button type="button" class="secondary-button small-button" data-review="${escapeHtml(batch.batch_id)}">Review contents</button>
              <button type="button" class="primary-button small-button" data-import="${escapeHtml(batch.batch_id)}">${actionLabel}</button>
            </div>
          </td>
        </tr>
      `;
    }).join("");
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
    byId("reviewImportBtn").disabled = false;
    byId("reviewImportBtn").textContent = imported ? "Archive Legacy Copy" : partiallyImported ? "Import Remaining & Archive" : "Import & Archive";
    byId("reviewDeleteBtn").disabled = imported || partiallyImported;

    if (imported) {
      setStatus(byId("migrationReviewStatus"), {
        title: "Legacy copy already imported",
        message: "Canonical runs already exist. Use Archive Legacy Copy to move the old folder into the migration archive.",
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
      <article class="record-card" data-claim-card="${escapeHtml(claim.claim_id)}">
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
        ${claim.has_report ? `
        <div class="inline-actions">
          <button type="button" class="secondary-button small-button"
            data-preview-claim="${escapeHtml(claim.claim_id)}"
            data-preview-root="${escapeHtml(claim.source_root || "")}">Preview report</button>
        </div>` : ""}
        <div class="migration-report-preview hidden" data-preview-for="${escapeHtml(claim.claim_id)}"></div>
      </article>
    `).join("") || `<div class="empty-state"><strong>No claim preview available.</strong></div>`;
  }

  async function previewReport(button) {
    const card = button.closest("[data-claim-card]");
    const container = card ? card.querySelector("[data-preview-for]") : null;
    if (!container) return;
    const claimId = button.dataset.previewClaim;
    const root = button.dataset.previewRoot || "";
    const batchId = selectedBatchId;
    if (!batchId) return;
    button.disabled = true;
    const originalLabel = button.textContent;
    button.textContent = "Loading…";
    try {
      const query = root ? `?root=${encodeURIComponent(root)}` : "";
      const data = await fetchJson(
        `/api/v1/migration/batches/${encodeURIComponent(batchId)}/claims/${encodeURIComponent(claimId)}/report${query}`
      );
      revealPanel(container);
      if (!data.has_report || !data.report_preview) {
        container.innerHTML = `<div class="empty-state"><strong>No report available for this claim yet.</strong></div>`;
      } else {
        const preview = data.report_preview;
        const evidence = preview.evidence || {};
        const rating = (preview.rating === null || preview.rating === undefined) ? "—" : String(preview.rating);
        container.innerHTML = `
          <div class="report-preview-grid">
            <div class="summary-cell"><span class="label">Rating</span><span class="value">${escapeHtml(rating)}</span></div>
            <div class="summary-cell"><span class="label">Relevant</span><span class="value">${Number(evidence.relevant || 0)}</span></div>
            <div class="summary-cell"><span class="label">Non-relevant</span><span class="value">${Number(evidence.non_relevant || 0)}</span></div>
            <div class="summary-cell"><span class="label">Inaccessible</span><span class="value">${Number(evidence.inaccessible || 0)}</span></div>
          </div>
          <p class="report-preview-explanation">${escapeHtml(preview.explanation || "")}</p>
        `;
      }
    } catch (error) {
      revealPanel(container);
      container.innerHTML = `<div class="status-card error-card"><strong>Could not load report preview.</strong><span>${escapeHtml(error.message)}</span></div>`;
    } finally {
      button.disabled = false;
      button.textContent = originalLabel;
    }
  }

  async function loadBatches() {
    const data = await fetchJson("/api/v1/migration/batches");
    migrationBatches = data.batches || [];
    renderTable();
  }

  async function reviewBatch(batchId) {
    const detail = await fetchJson(`/api/v1/migration/batches/${encodeURIComponent(batchId)}`);
    renderReview(detail);
    const panel = byId("migrationReviewPanel");
    if (panel && typeof panel.scrollIntoView === "function") {
      panel.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  async function importBatch(batchId, archiveAfter = false) {
    const data = await fetchJson(`/api/v1/migration/batches/${encodeURIComponent(batchId)}/import`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ archive_after: archiveAfter }),
    });
    const archived = Boolean(data.archive?.archived);
    const createdCount = (data.runs || []).filter(run => !run.already_imported).length;
    setStatus(byId("migrationStatus"), {
      title: archived && !createdCount ? "Legacy copy archived" : archiveAfter ? "Batch imported and archived" : "Batch imported",
      message: archived
        ? `${batchId} has canonical runs and its legacy folder was moved to the migration archive.`
        : `${batchId} has been imported into the claim store.`,
      tone: "success",
    });
    await loadBatches();
    if (archived) {
      clearReview();
      return;
    }
    await reviewBatch(batchId).catch(() => clearReview());
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
    clearReview();
    await loadBatches();
  }

  async function importAll() {
    const data = await fetchJson("/api/v1/migration/import_all", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ archive_after: true }),
    });
    const batchCount = data.batch_count || 0;
    const remaining = data.remaining_pending_count || 0;
    if (!batchCount) {
      setStatus(byId("migrationStatus"), {
        title: "No pending legacy batches",
        message: "All detected legacy folders already have canonical runs or there are no legacy folders to migrate.",
        tone: "info",
      });
      await loadBatches();
      return;
    }
    setStatus(byId("migrationStatus"), {
      title: remaining ? "Some pending batches remain" : "Pending legacy batches migrated",
      message: `${batchCount} batch${batchCount === 1 ? "" : "es"} processed; ${data.created_count || 0} new run${data.created_count === 1 ? "" : "s"} created; ${data.already_imported_count || 0} already existed; ${data.archived_count || 0} legacy folder${data.archived_count === 1 ? "" : "s"} archived.`,
      tone: remaining ? "warning" : "success",
    });
    clearReview();
    await loadBatches();
  }

  byId("refreshMigrationBtn").addEventListener("click", () => {
    const button = byId("refreshMigrationBtn");
    const restore = buttonBusy(button, "Refreshing…");
    loadBatches().then(() => {
      restore();
      flashButton(button, { label: "Refreshed ✓", duration: 1200 });
    }).catch(error => {
      restore();
      flashButton(button, { label: "Refresh failed ✗", tone: "error" });
      setStatus(byId("migrationStatus"), { title: "Refresh failed", message: error.message, tone: "error" });
    });
  });
  byId("importAllBtn").addEventListener("click", () => {
    const button = byId("importAllBtn");
    const restore = buttonBusy(button, "Migrating…");
    importAll().then(() => {
      restore();
      flashButton(button, { label: "Done ✓" });
    }).catch(error => {
      restore();
      flashButton(button, { label: "Import failed ✗", tone: "error" });
      setStatus(byId("migrationStatus"), { title: "Import failed", message: error.message, tone: "error" });
    });
  });
  byId("migrationTableBody").addEventListener("click", event => {
    const reviewButton = event.target.closest("[data-review]");
    if (reviewButton) {
      const restore = buttonBusy(reviewButton, "Loading…");
      reviewBatch(reviewButton.dataset.review).then(restore).catch(error => {
        restore();
        flashButton(reviewButton, { label: "Failed ✗", tone: "error" });
        setStatus(byId("migrationStatus"), { title: "Review failed", message: error.message, tone: "error" });
      });
      return;
    }
    const importButton = event.target.closest("[data-import]");
    if (importButton) {
      const restore = buttonBusy(importButton, "Importing…");
      importBatch(importButton.dataset.import, true).then(restore).catch(error => {
        restore();
        flashButton(importButton, { label: "Failed ✗", tone: "error" });
        setStatus(byId("migrationStatus"), { title: "Import failed", message: error.message, tone: "error" });
      });
    }
  });

  byId("migrationReviewContent").addEventListener("click", event => {
    const previewButton = event.target.closest("[data-preview-claim]");
    if (previewButton) {
      previewReport(previewButton).catch(error => setStatus(byId("migrationReviewStatus"), { title: "Preview failed", message: error.message, tone: "error" }));
    }
  });

  byId("reviewImportBtn").addEventListener("click", () => {
    if (!selectedBatchId) {
      return;
    }
    const button = byId("reviewImportBtn");
    const restore = buttonBusy(button, "Importing…");
    importBatch(selectedBatchId, true).then(() => {
      restore();
      // importBatch re-renders the review actions with a fresh label; restore()
      // put the stale pre-click label back, so re-sync from the current detail.
      if (selectedBatchDetail) renderReviewActions(selectedBatchDetail);
    }).catch(error => {
      restore();
      flashButton(button, { label: "Import failed ✗", tone: "error" });
      setStatus(byId("migrationStatus"), { title: "Import failed", message: error.message, tone: "error" });
    });
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
