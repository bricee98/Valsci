(() => {
  const { escapeHtml, fetchJson, formatCurrency, formatDateTime } = window.ValsciUI;
  const byId = (id) => document.getElementById(id);

  function claimHref(claim) {
    const latestRun = claim.latest_run;
    const suffix = latestRun?.run_id ? `?run_id=${encodeURIComponent(latestRun.run_id)}` : "";
    return `/claims/${encodeURIComponent(claim.claim_key)}${suffix}`;
  }

  function renderClaims(claims) {
    byId("claimCount").textContent = claims.length;
    const target = byId("claimList");
    if (!claims.length) {
      target.innerHTML = `<div class="empty-state"><strong>No claims found.</strong><span>Try a broader search term or queue a new run from Home.</span></div>`;
      return;
    }

    target.innerHTML = claims.map(claim => {
      const latestRun = claim.latest_run;
      return `
        <article class="record-card">
          <div class="stack">
            <strong>${escapeHtml(claim.text)}</strong>
            <div class="record-meta">
              <span>${claim.run_count} run${claim.run_count === 1 ? "" : "s"}</span>
              ${claim.batch_tags?.length ? `<span>${escapeHtml(claim.batch_tags.join(", "))}</span>` : ""}
              ${latestRun?.last_activity_at ? `<span>${escapeHtml(formatDateTime(latestRun.last_activity_at))}</span>` : ""}
            </div>
            ${latestRun ? `
              <div class="pill-row">
                <span class="pill">${escapeHtml(latestRun.rating_label || latestRun.status)}</span>
                <span class="pill">${escapeHtml(latestRun.provider_label || latestRun.default_model || "Unknown provider")}</span>
                <span class="pill">${escapeHtml(latestRun.candidate_prefix ? `${latestRun.candidate_prefix} · ${latestRun.candidate_label}` : (latestRun.current_stage_label || latestRun.status))}</span>
                <span class="pill">${escapeHtml(formatCurrency((latestRun.usage || {}).cost_usd || 0))}</span>
              </div>
            ` : `<p class="helper-text">No runs are attached to this claim yet.</p>`}
          </div>
          <div class="inline-actions">
            <a href="${claimHref(claim)}" class="primary-button small-button">Open Claim</a>
          </div>
        </article>
      `;
    }).join("");
  }

  async function loadClaims() {
    const query = byId("searchInput").value.trim();
    const data = await fetchJson(`/api/v1/claims?search=${encodeURIComponent(query)}`);
    renderClaims(data.claims || []);
  }

  byId("searchBtn").addEventListener("click", () => loadClaims().catch(error => alert(error.message)));
  byId("clearBtn").addEventListener("click", () => {
    byId("searchInput").value = "";
    loadClaims().catch(error => alert(error.message));
  });
  byId("searchInput").addEventListener("keydown", event => {
    if (event.key === "Enter") {
      loadClaims().catch(error => alert(error.message));
    }
  });

  loadClaims().catch(error => {
    byId("claimList").innerHTML = `<div class="status-card error-card"><strong>Claims failed to load.</strong><span>${escapeHtml(error.message)}</span></div>`;
  });
})();
