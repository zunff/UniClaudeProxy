/*
 * DeepSeek web driver (page-context init script).
 *
 * Injected by deepseek_web_provider.py via context.add_init_script() so it runs
 * in every new document. It wraps window.fetch and XMLHttpRequest and forwards
 * the raw SSE text of the page's own streaming requests to Python through the
 * exposed binding `window.__lxEmit(jsonString)`.
 *
 * Messages sent to Python (JSON strings):
 *   {"kind":"start","status":200}          response headers received
 *   {"kind":"chunk","text":"..."}          decoded SSE text (may split frames)
 *   {"kind":"end"}                         stream finished normally
 *   {"kind":"error","message":"..."}       transport failure / non-2xx status
 *
 * All SSE / JSON-patch parsing happens in Python so it can be unit tested.
 *
 * Debugging without the proxy:
 *   python -m app.providers.deepseek_web_provider --prompt "hello"
 */
(() => {
  if (window.__lxDriverInstalled) return;
  window.__lxDriverInstalled = true;

  const STREAM_PATHS = [
    '/api/v0/chat/completion',
    '/api/v0/chat/edit_message',
    '/api/v0/chat/regenerate',
  ];

  const isStreamUrl = (url) => {
    try {
      const path = new URL(String(url), location.href).pathname;
      return STREAM_PATHS.some((p) => path === p);
    } catch (_) {
      return false;
    }
  };

  const emit = (obj) => {
    try {
      const fn = window.__lxEmit;
      if (typeof fn === 'function') {
        // Account id lets one Python callback serve several browser contexts.
        obj.__account = window.__lxAccount || '';
        fn(JSON.stringify(obj));
      }
    } catch (_) {
      /* binding not ready or page is closing */
    }
  };

  // ---- fetch -------------------------------------------------------------
  const origFetch = window.fetch;
  window.fetch = async function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    const resp = await origFetch.apply(this, arguments);
    if (!isStreamUrl(url)) return resp;

    emit({ kind: 'start', status: resp.status });
    if (!resp.ok || !resp.body) {
      emit({ kind: 'error', message: 'HTTP ' + resp.status });
      return resp;
    }
    // Read a clone so the page's own consumer is unaffected.
    const reader = resp.clone().body.getReader();
    const decoder = new TextDecoder('utf-8');
    (async () => {
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          const text = decoder.decode(value, { stream: true });
          if (text) emit({ kind: 'chunk', text });
        }
        const tail = decoder.decode();
        if (tail) emit({ kind: 'chunk', text: tail });
        emit({ kind: 'end' });
      } catch (e) {
        emit({ kind: 'error', message: String((e && e.message) || e) });
      }
    })();
    return resp;
  };

  // ---- XMLHttpRequest ----------------------------------------------------
  const origOpen = XMLHttpRequest.prototype.open;
  const origSend = XMLHttpRequest.prototype.send;

  XMLHttpRequest.prototype.open = function (method, url) {
    this.__lxStream = isStreamUrl(url);
    return origOpen.apply(this, arguments);
  };

  XMLHttpRequest.prototype.send = function () {
    if (this.__lxStream) {
      let offset = 0;
      let started = false;
      const flush = () => {
        let text = '';
        try {
          text = this.responseText || '';
        } catch (_) {
          return;
        }
        if (text.length > offset) {
          emit({ kind: 'chunk', text: text.slice(offset) });
          offset = text.length;
        }
      };
      this.addEventListener('readystatechange', () => {
        if (this.readyState >= 2 && !started) {
          started = true;
          emit({ kind: 'start', status: this.status });
        }
      });
      this.addEventListener('progress', flush);
      this.addEventListener('loadend', () => {
        flush();
        if (this.status >= 200 && this.status < 300) emit({ kind: 'end' });
        else emit({ kind: 'error', message: 'HTTP ' + this.status });
      });
    }
    return origSend.apply(this, arguments);
  };
})();
