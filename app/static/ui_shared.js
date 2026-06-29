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

  const ratingMeta = {
    0: { label: "No Evidence", tone: "rating-0" },
    1: { label: "Contradicted", tone: "rating-1" },
    2: { label: "Likely False", tone: "rating-2" },
    3: { label: "Mixed Evidence", tone: "rating-3" },
    4: { label: "Likely True", tone: "rating-4" },
    5: { label: "Highly Supported", tone: "rating-5" },
  };

  function ratingLabel(rating) {
    return ratingMeta[rating]?.label || "Unrated";
  }

  function ratingTone(rating) {
    return ratingMeta[rating]?.tone || "rating-none";
  }

  function ratingChip(rating, { label, compact = false, extraClass = "", title } = {}) {
    const meta = ratingMeta[rating];
    const text = label || meta?.label || "Unrated";
    const display = meta && !compact ? `${rating} · ${text}` : text;
    const tooltip = title || (meta ? `${rating} — ${text} (0–5 scale)` : text);
    return `<span class="rating-chip ${ratingTone(rating)} ${extraClass}" title="${escapeHtml(tooltip)}">${escapeHtml(display)}</span>`;
  }

  function formatTokens(value) {
    const tokens = Number(value ?? 0);
    if (!Number.isFinite(tokens) || tokens <= 0) {
      return "0";
    }
    if (tokens < 1000) {
      return String(Math.round(tokens));
    }
    if (tokens < 1000000) {
      return `${(tokens / 1000).toFixed(1)}k`;
    }
    return `${(tokens / 1000000).toFixed(2)}M`;
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || `Request failed for ${url}`);
    }
    return data;
  }

  function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    const viewHeight = window.innerHeight || document.documentElement.clientHeight;
    if (rect.height === 0 && rect.width === 0) {
      return false;
    }
    return rect.bottom > 0 && rect.top < viewHeight;
  }

  function toastStack() {
    let stack = document.getElementById("valsciToastStack");
    if (!stack) {
      stack = document.createElement("div");
      stack.id = "valsciToastStack";
      stack.className = "toast-stack";
      stack.setAttribute("aria-live", "polite");
      document.body.appendChild(stack);
    }
    return stack;
  }

  function dismissToast(toast) {
    if (!toast.isConnected || toast.classList.contains("toast-leaving")) {
      return;
    }
    toast.classList.add("toast-leaving");
    window.setTimeout(() => toast.remove(), 250);
  }

  function showToast({ title = "", message = "", tone = "info", duration = 6000 }) {
    const toast = document.createElement("div");
    toast.className = `toast status-card ${tone}-card`;
    toast.setAttribute("role", tone === "error" ? "alert" : "status");
    toast.innerHTML = `
      ${title ? `<strong>${escapeHtml(title)}</strong>` : ""}
      ${message ? `<span>${escapeHtml(message)}</span>` : ""}
    `;
    toast.title = "Dismiss";
    toast.addEventListener("click", () => dismissToast(toast));
    toastStack().appendChild(toast);
    window.setTimeout(() => dismissToast(toast), duration);
    return toast;
  }

  // toast: "auto" mirrors the message into a fixed toast only when the status
  // card is scrolled out of view, so feedback is visible no matter where the
  // user is on the page. Pass toast: false for ambient/load-time statuses.
  function setStatus(target, { title = "", message = "", tone = "info", toast = "auto" }) {
    if (!target) {
      if (toast) {
        showToast({ title, message, tone });
      }
      return;
    }
    target.className = `status-card ${tone}-card`;
    target.innerHTML = `
      ${title ? `<strong>${escapeHtml(title)}</strong>` : ""}
      ${message ? `<span>${escapeHtml(message)}</span>` : ""}
    `;
    target.classList.remove("hidden");
    target.classList.remove("status-pulse");
    void target.offsetWidth;
    target.classList.add("status-pulse");
    if (toast === true || (toast === "auto" && !isInViewport(target))) {
      showToast({ title, message, tone });
    }
  }

  function revealPanel(element, { block = "nearest" } = {}) {
    if (!element) {
      return;
    }
    element.classList.remove("hidden");
    window.requestAnimationFrame(() => {
      element.scrollIntoView({ behavior: "smooth", block });
      element.classList.remove("panel-flash");
      void element.offsetWidth;
      element.classList.add("panel-flash");
    });
  }

  function flashButton(button, { label, tone = "success", duration = 1800 } = {}) {
    if (!button) {
      return;
    }
    if (button.dataset.flashRestore === undefined) {
      button.dataset.flashRestore = button.textContent;
    }
    if (button.dataset.flashTimer) {
      window.clearTimeout(Number(button.dataset.flashTimer));
    }
    button.textContent = label || (tone === "error" ? "Failed ✗" : "Saved ✓");
    button.classList.remove("button-flash-success", "button-flash-error");
    button.classList.add(tone === "error" ? "button-flash-error" : "button-flash-success");
    const timer = window.setTimeout(() => {
      button.textContent = button.dataset.flashRestore;
      delete button.dataset.flashRestore;
      delete button.dataset.flashTimer;
      button.classList.remove("button-flash-success", "button-flash-error");
    }, duration);
    button.dataset.flashTimer = String(timer);
  }

  function buttonBusy(button, busyLabel = "Working…") {
    if (!button) {
      return () => {};
    }
    const original = button.textContent;
    const wasDisabled = button.disabled;
    button.textContent = busyLabel;
    button.disabled = true;
    button.classList.add("button-busy");
    return () => {
      button.textContent = original;
      button.disabled = wasDisabled;
      button.classList.remove("button-busy");
    };
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
                  <th
                    class="candidate-header"
                    style="${candidateStyle({ color: item._candidateColor })}"
                    ${detail ? 'role="button" tabindex="0" aria-expanded="false"' : ""}
                  >${rowHeader(item, index)}</th>
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
    formatTokens,
    hideStatus,
    setStatus,
    showToast,
    revealPanel,
    flashButton,
    buttonBusy,
    isInViewport,
    stageLabel,
    ratingLabel,
    ratingTone,
    ratingChip,
    candidateStyle,
    renderTransposedTable,
  };

  function visibleModal() {
    const modals = Array.from(document.querySelectorAll(".modal:not(.hidden)"));
    return modals.length ? modals[modals.length - 1] : null;
  }

  function focusModal(modal) {
    if (!modal) return;
    const focusable = modal.querySelector(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    (focusable || modal).focus();
  }

  function closeModal(modal) {
    if (modal) modal.classList.add("hidden");
  }

  function enhanceModals() {
    document.querySelectorAll(".modal").forEach((modal) => {
      modal.setAttribute("role", "dialog");
      modal.setAttribute("aria-modal", "true");
      modal.setAttribute("tabindex", "-1");
      const title = modal.querySelector("h1, h2, h3");
      if (title && !modal.getAttribute("aria-labelledby")) {
        if (!title.id) {
          title.id = `modal-title-${Math.random().toString(36).slice(2)}`;
        }
        modal.setAttribute("aria-labelledby", title.id);
      }
      const observer = new MutationObserver(() => {
        if (!modal.classList.contains("hidden")) {
          window.setTimeout(() => focusModal(modal), 0);
        }
      });
      observer.observe(modal, { attributes: true, attributeFilter: ["class"] });
      modal.addEventListener("click", (event) => {
        if (event.target === modal) {
          closeModal(modal);
        }
      });
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeModal(visibleModal());
    }
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", enhanceModals);
  } else {
    enhanceModals();
  }
})();
