// === FlexViz mouse-wheel mode ===
// The spec sets the page's mode (toolbar.wheel_mode). A viewer's choice is a
// preference like the light/dark mode: localStorage only, never the spec or a
// share URL.
(function() {
  const MODE_KEY = 'fv-wheel';
  const PREFERENCES = ['auto', 'zoom', 'ctrl', 'off'];
  const KEY = /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘' : 'Ctrl';
  const LABELS = { auto: 'Auto', zoom: 'Zoom', ctrl: KEY, off: 'Off' };
  const HINT_MS = 1500;
  const button = document.getElementById('fv-btn-wheel');
  let preference = DASHBOARD_SPEC.layout.toolbar.wheel_mode;
  // Without the button the viewer cannot see or change a stored choice, so
  // the page's mode applies.
  if (button) {
    try {
      const stored = window.localStorage.getItem(MODE_KEY);
      if (PREFERENCES.includes(stored)) preference = stored;
    } catch (e) { /* storage blocked: the page's mode */ }
  }

  window.fvWheelZooms = function(event) {
    // In an iframe (a notebook, a web app) a plain wheel must scroll the page.
    // A trackpad pinch arrives as Ctrl + wheel.
    const mode = preference === 'auto'
      ? (window.self === window.top ? 'zoom' : 'ctrl')
      : preference;
    if (mode !== 'ctrl') return mode === 'zoom';
    if (event.ctrlKey || event.metaKey) return true;
    showHint(event.target.closest('.fv-plot-wrap'));
    return false;
  };

  let hint = null;
  let hintTimer = 0;
  function showHint(plotWrap) {
    if (!plotWrap) return;
    if (!hint) {
      hint = document.createElement('div');
      hint.className = 'fv-wheel-hint';
      hint.textContent = `Use ${KEY} + scroll to zoom`;
    }
    if (hint.parentNode !== plotWrap) plotWrap.appendChild(hint);
    hint.classList.add('visible');
    clearTimeout(hintTimer);
    hintTimer = setTimeout(hideHint, HINT_MS);
  }
  function hideHint() {
    clearTimeout(hintTimer);
    hint?.classList.remove('visible');
  }

  if (!button) return;
  const label = () => { button.textContent = 'Wheel: ' + LABELS[preference]; };
  label();
  button.addEventListener('click', () => {
    preference = PREFERENCES[(PREFERENCES.indexOf(preference) + 1) % PREFERENCES.length];
    // Store Auto too: it can differ from the mode of another page.
    try {
      window.localStorage.setItem(MODE_KEY, preference);
    } catch (e) { /* storage blocked: the choice lasts for this page only */ }
    label();
    // A shown hint names the key of the old mode.
    hideHint();
  });
})();
