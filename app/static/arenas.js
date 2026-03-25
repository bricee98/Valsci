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

  function renderArenas() {
    const arenas = allArenas.filter(matchesFilters);
    const target = byId("arenaLibrary");
    if (!arenas.length) {
      target.innerHTML = `<div class="empty-state"><strong>No arenas match the current filter.</strong><span>Start a new arena from the page action to see it here.</span></div>`;
      return;
    }

    target.innerHTML = arenas.map(arena => `
      <article class="arena-card">
        <div class="panel-header">
          <div>
            <h2 class="panel-title">${escapeHtml(arena.title)}</h2>
            <p class="panel-subtitle">${escapeHtml(arena.arena_id)} · ${escapeHtml(arena.current_stage_label || "")}</p>
          </div>
          <span class="badge ${arena.status === "completed" ? "success-badge" : "neutral-badge"}">${escapeHtml(arena.status.replace(/_/g, " "))}</span>
        </div>
        <div class="arena-meta">
          <span>${arena.claim_count} claim${arena.claim_count === 1 ? "" : "s"}</span>
          <span>${arena.candidate_count} candidate${arena.candidate_count === 1 ? "" : "s"}</span>
          <span>Expected ${escapeHtml(formatCurrency(arena.expected_cost_usd))}</span>
          <span>Actual ${escapeHtml(formatCurrency(arena.actual_cost_usd))}</span>
          <span>Updated ${escapeHtml(formatDateTime(arena.updated_at))}</span>
        </div>
        <div class="inline-actions">
          <a href="/arena_results?arena_id=${encodeURIComponent(arena.arena_id)}" class="primary-button small-button">Open Workspace</a>
        </div>
      </article>
    `).join("");
  }

  async function loadArenas() {
    const data = await fetchJson("/api/v1/arenas");
    allArenas = data.arenas || [];
    renderArenas();
  }

  byId("arenaSearchInput").addEventListener("input", renderArenas);
  byId("arenaStatusFilter").addEventListener("change", renderArenas);

  loadArenas().catch(error => {
    byId("arenaLibrary").innerHTML = `<div class="status-card error-card"><strong>Arenas failed to load.</strong><span>${escapeHtml(error.message)}</span></div>`;
  });
})();
