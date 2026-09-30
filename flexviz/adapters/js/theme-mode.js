// === FlexViz light/dark mode (runs in <head>, before the first paint) ===
// The mode is a viewer preference, not interaction state: it lives in
// localStorage and never in the spec or a share URL. 'auto' follows the OS.
// A change calls window.fvApplyTheme, which the renderer bundle defines.
(function() {
  const root = document.documentElement;
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  const MODE_KEY = 'fv-mode';
  const PREFERENCES = ['auto', 'light', 'dark'];
  let preference = 'auto';
  try {
    const stored = window.localStorage.getItem(MODE_KEY);
    if (stored === 'light' || stored === 'dark') preference = stored;
  } catch (e) { /* storage blocked: follow the OS */ }

  function applyMode() {
    const mode = preference === 'auto' ? (media.matches ? 'dark' : 'light') : preference;
    if (root.dataset.fvMode === mode) return;
    root.dataset.fvMode = mode;
    window.fvApplyTheme?.();
  }
  applyMode();
  media.addEventListener('change', applyMode);

  function labelButton(button) {
    button.textContent = 'Mode: ' + preference.charAt(0).toUpperCase() + preference.slice(1);
  }

  document.addEventListener('DOMContentLoaded', () => {
    const button = document.getElementById('fv-btn-mode');
    if (!button) return;
    labelButton(button);
    button.addEventListener('click', () => {
      preference = PREFERENCES[(PREFERENCES.indexOf(preference) + 1) % PREFERENCES.length];
      try {
        if (preference === 'auto') window.localStorage.removeItem(MODE_KEY);
        else window.localStorage.setItem(MODE_KEY, preference);
      } catch (e) { /* storage blocked: the choice lasts for this page only */ }
      labelButton(button);
      applyMode();
    });
  });
})();
