// === FlexViz shared runtime — stream polling ===
// Requires: delta.js loaded first (postDashboardUpdate, SERVER_URL).
//
// A changed stream version triggers one refresh request: it recomputes every
// trace and writes no viewport, so autorange axes follow the data and zoomed
// axes keep their range.

const _fvStreamVersions = { ...FV_STREAMING_SOURCES };

// A refresh redraws the figure under the pointer, so none starts during a
// drag or brush.
let _fvPointerDown = false;
document.addEventListener('pointerdown', () => { _fvPointerDown = true; }, true);
window.addEventListener('pointerup', () => { _fvPointerDown = false; });
window.addEventListener('pointercancel', () => { _fvPointerDown = false; });

async function _fvPollStreams() {
  if (!document.hidden && !_fvPointerDown) {
    try {
      const changed = {};
      for (const name of Object.keys(_fvStreamVersions)) {
        const resp = await fetch(SERVER_URL + '/sources/' + encodeURIComponent(name) + '/version');
        if (!resp.ok) continue;
        const version = await resp.json();
        if (version !== _fvStreamVersions[name]) changed[name] = version;
      }
      // Keep the versions only after a refresh that worked, so a failed one
      // is retried on the next poll.
      if (Object.keys(changed).length && !_fvPointerDown
          && await postDashboardUpdate({ type: 'refresh', force_update: true })) {
        Object.assign(_fvStreamVersions, changed);
      }
    } catch (e) {
      console.warn('flexviz stream poll failed', e);
    }
  }
  // The next poll starts after this one ends, so polls never overlap.
  window.setTimeout(_fvPollStreams, 1000);
}

if (Object.keys(_fvStreamVersions).length) window.setTimeout(_fvPollStreams, 1000);
