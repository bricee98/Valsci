(() => {
  const stageLabels = {
    query_generation: "Query Generation",
    paper_analysis: "Review Evidence",
    venue_scoring: "Review Scores",
    final_report: "Final Reports",
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatCurrency(value) {
    const amount = Number(value ?? 0);
    return `$${amount.toFixed(4)}`;
  }

  function formatShortCurrency(value) {
    const amount = Number(value ?? 0);
    return `$${amount.toFixed(2)}`;
  }

  function formatDurationMs(value) {
    const totalMs = Number(value ?? 0);
    if (!Number.isFinite(totalMs) || totalMs <= 0) {
      return "0s";
    }
    const totalSeconds = Math.round(totalMs / 1000);
    if (totalSeconds < 60) {
      return `${totalSeconds}s`;
    }
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    if (minutes < 60) {
      return seconds ? `${minutes}m ${seconds}s` : `${minutes}m`;
    }
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return remainingMinutes ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
  }

  function formatDateTime(value) {
    if (!value) {
      return "Unknown";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return String(value);
    }
    return date.toLocaleString([], {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  }

  function stageLabel(stage) {
    return stageLabels[stage] || stage || "Unknown";
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || `Request failed for ${url}`);
    }
    return data;
  }

  function setStatus(target, { title = "", message = "", tone = "info" }) {
    if (!target) {
      return;
    }
    target.className = `status-card ${tone}-card`;
    target.innerHTML = `
      ${title ? `<strong>${escapeHtml(title)}</strong>` : ""}
      ${message ? `<span>${escapeHtml(message)}</span>` : ""}
    `;
    target.classList.remove("hidden");
  }

  function hideStatus(target) {
    if (!target) {
      return;
    }
    target.classList.add("hidden");
    target.innerHTML = "";
  }

  function candidateStyle(candidate) {
    const color = candidate?.color || "#0f766e";
    return `--candidate-color:${escapeHtml(color)}`;
  }

  function renderTransposedTable({ items, columns, rowHeader, detailContent, focusTest, tableClass }) {
    if (!items.length) return "";
    const colCount = columns.length + 1;
    const hasDetails = typeof detailContent === "function";
    return `
      <div class="comparison-scroll">
        <table class="comparison-table ${tableClass || ""}">
          <thead>
            <tr>
              <th></th>
              ${columns.map((col) => `<th>${escapeHtml(col.label)}</th>`).join("")}
            </tr>
          </thead>
          <tbody>
            ${items.map((item, index) => {
              const isFocused = focusTest ? focusTest(item) : false;
              const detail = hasDetails ? detailContent(item) : null;
              return `
                <tr class="${isFocused ? "focused-row" : ""}">
                  <th class="candidate-header" style="${candidateStyle({ color: item._candidateColor })}">${rowHeader(item, index)}</th>
                  ${columns.map((col) => {
                    const value = col.cell(item);
                    const highlighted = col.highlight ? col.highlight(item, items) : false;
                    return `<td class="${highlighted ? "metric-highlight" : ""}">${value}</td>`;
                  }).join("")}
                </tr>
                ${detail ? `<tr class="detail-row"><td colspan="${colCount}"><div class="detail-cards">${detail}</div></td></tr>` : ""}
              `;
            }).join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  window.ValsciUI = {
    escapeHtml,
    fetchJson,
    formatCurrency,
    formatShortCurrency,
    formatDateTime,
    formatDurationMs,
    hideStatus,
    setStatus,
    stageLabel,
    candidateStyle,
    renderTransposedTable,
  };
})();
