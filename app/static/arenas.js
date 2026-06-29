(() => {
  const { escapeHtml, fetchJson, formatCurrency, formatDateTime } = window.ValsciUI;
  const byId = (id) => document.getElementById(id);
  let allArenas = [];

  function matchesFilters(arena) {
    const query = byId("arenaSearchInput").value.trim().toLowerCase();
    const statusFilter = byId("arenaStatusFilter").value;
    const haystack = [arena.title, arena.arena_id].join(" ").toLowerCase();
    if (query && !haystack.includes(query)) {
      return false;
    }
    if (statusFilter && arena.status !== statusFilter) {
      return false;
    }
    return true;
  }

  function statusBadge(arena) {
    if (arena.status === "completed") {
      return { tone: "success-badge", label: "completed" };
    }
    if (arena.status === "ready_for_review") {
      return { tone: "warning-badge", label: "ready for review" };
    }
    if (arena.status === "needs_attention") {
      return { tone: "error-badge", label: "needs attention" };
    }
    return { tone: "neutral-badge", label: String(arena.status || "").replace(/_/g, " ") };
  }

  function runProgress(arena) {
    const counts = arena.status_counts || {};
    const total = Object.values(counts).reduce((sum, value) => sum + Number(value || 0), 0);
    if (!total) {
      return "";
    }
    const done = Number(counts.processed || 0);
    const failed = Number(counts.error || 0) + Number(counts.failed || 0);
    const failedNote = failed ? ` · ${failed} failed` : "";
    return `<span>${done}/${total} runs done${failedNote}</span>`;
  }

  function renderArenas() {
    const arenas = allArenas.filter(matchesFilters);
    const target = byId("arenaLibrary");
    if (!arenas.length) {
      const hasFilter = byId("arenaSearchInput").value.trim() || byId("arenaStatusFilter").value;
      target.innerHTML = hasFilter
        ? `<div class="empty-state"><strong>No arenas match the current filter.</strong><span>Clear the search or status filter to see all arenas.</span></div>`
        : `<div class="empty-state"><strong>No arenas yet.</strong><span>Arenas run the same claims through multiple models so you can compare their outputs side by side.</span><a href="/arena" class="primary-button small-button" style="margin-top:10px">New Arena</a></div>`;
      return;
    }

    target.innerHTML = arenas.map(arena => {
      const badge = statusBadge(arena);
      return `
      <article class="arena-card">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">${escapeHtml(arena.title)}</h2>
            <p class="panel-subtitle">${escapeHtml(arena.arena_id)} · ${escapeHtml(arena.current_stage_label || "")}</p>
          </div>
          <span class="badge ${badge.tone}">${escapeHtml(badge.label)}</span>
        </div>
        <div class="arena-meta">
          <span>${arena.claim_count} claim${arena.claim_count === 1 ? "" : "s"}</span>
          <span>${arena.candidate_count} model${arena.candidate_count === 1 ? "" : "s"}</span>
          ${runProgress(arena)}
          <span>Expected ${escapeHtml(formatCurrency(arena.expected_cost_usd))}</span>
          <span>Actual ${escapeHtml(formatCurrency(arena.actual_cost_usd))}</span>
          <span>Updated ${escapeHtml(formatDateTime(arena.updated_at))}</span>
        </div>
        <div class="inline-actions">
          <a href="/arena_results?arena_id=${encodeURIComponent(arena.arena_id)}" class="primary-button small-button">${arena.status === "ready_for_review" ? "Review & Continue" : arena.status === "needs_attention" ? "Review Failures" : "Open Workspace"}</a>
        </div>
      </article>
    `;
    }).join("");
  }

  async function loadArenas() {
    const data = await fetchJson("/api/v1/arenas");
    allArenas = data.arenas || [];
    renderArenas();
  }

  byId("arenaSearchInput").addEventListener("input", renderArenas);
  byId("arenaStatusFilter").addEventListener("change", renderArenas);

  byId("arenaLibrary").innerHTML = `<div class="empty-state"><strong>Loading arenas…</strong></div>`;
  loadArenas().catch(error => {
    byId("arenaLibrary").innerHTML = `<div class="status-card error-card"><strong>Arenas failed to load.</strong><span>${escapeHtml(error.message)}</span></div>`;
  });
})();
