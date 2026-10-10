// === FlexViz mouse-wheel mode ===
// A viewer preference like the light/dark mode: localStorage only, never the
// spec or a share URL.
(function() {
  const MODE_KEY = 'fv-wheel';
  const PREFERENCES = ['auto', 'zoom', 'ctrl', 'off'];
  const KEY = /Mac|iPhone|iPad/.test(navigator.platform) ? '⌘' : 'Ctrl';
  const LABELS = { auto: 'Auto', zoom: 'Zoom', ctrl: KEY, off: 'Off' };
  const HINT_MS = 1500;
  let preference = 'auto';
  try {
    const stored = window.localStorage.getItem(MODE_KEY);
    if (PREFERENCES.includes(stored)) preference = stored;
  } catch (e) { /* storage blocked: auto */ }

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

  const button = document.getElementById('fv-btn-wheel');
  if (!button) return;
  const label = () => { button.textContent = 'Wheel: ' + LABELS[preference]; };
  label();
  button.addEventListener('click', () => {
    preference = PREFERENCES[(PREFERENCES.indexOf(preference) + 1) % PREFERENCES.length];
    try {
      if (preference === 'auto') window.localStorage.removeItem(MODE_KEY);
      else window.localStorage.setItem(MODE_KEY, preference);
    } catch (e) { /* storage blocked: the choice lasts for this page only */ }
    label();
    // A shown hint names the key of the old mode.
    hideHint();
  });
})();
