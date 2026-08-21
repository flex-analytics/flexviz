// === <fv-panel> shared structure + control binding helpers ===
// Python emits the full panel HTML. This component only normalizes the DOM so
// every renderer gets the same `.fv-plot-wrap` structure.

class FvPanel extends HTMLElement {
  connectedCallback() {
    if (this.dataset.fvPanelReady === 'true') return;
    this.dataset.fvPanelReady = 'true';
    this.classList.add('fv-panel');

    for (const child of Array.from(this.children)) {
      if (child.classList && child.classList.contains('fv-plot-wrap')) return;
    }

    const plotWrap = document.createElement('div');
    plotWrap.className = 'fv-plot-wrap';
    while (this.firstChild) {
      plotWrap.appendChild(this.firstChild);
    }
    this.appendChild(plotWrap);
  }
}

function bindPanelButton(button, handler) {
  if (!button || button.dataset.fvBound === 'true') return;
  button.dataset.fvBound = 'true';
  button.addEventListener('click', function(event) {
    event.stopPropagation();
    handler.call(this, event);
  });
}

window.fvPanelControlRoot = function(figIdx) {
  return document.getElementById('fv-bar-' + figIdx);
};

window.fvBindPanelControls = function(figUid, handlers) {
  if (!figUid || typeof figUidToIdx === 'undefined') return;
  const figIdx = figUidToIdx[figUid];
  if (figIdx === undefined) return;
  const controls = window.fvPanelControlRoot?.(figIdx);
  if (!controls) return;

  for (const btn of controls.querySelectorAll('.fv-mode-btn')) {
    bindPanelButton(btn, function() {
      handlers?.setMode?.(this.dataset.mode);
    });
  }
  for (const btn of controls.querySelectorAll('.fv-mode-action-btn[data-action="reset-panel"]')) {
    bindPanelButton(btn, function() {
      handlers?.resetPanel?.();
    });
  }
  for (const btn of controls.querySelectorAll('.fv-mode-action-btn[data-action="lock-axes"]')) {
    bindPanelButton(btn, function() {
      handlers?.toggleAxisLocks?.();
    });
  }
};

if (!customElements.get('fv-panel')) {
  customElements.define('fv-panel', FvPanel);
}
