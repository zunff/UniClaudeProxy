/*
 * Generic web-chat driver (page-context init script).
 *
 * Injected by the browser engine (web/engine.py) via context.add_init_script()
 * so it runs in every new document. It wraps window.fetch and XMLHttpRequest and
 * forwards the raw text of the page's own streaming requests to Python through
 * the exposed binding `window.__lxEmit(jsonString)`.
 *
 * The page is configured before this script runs:
 *   window.__lxSite         provider type of this browser (e.g. "deepseek-web")
 *   window.__lxAccount      account id driving this browser ("" = default)
 *   window.__lxStreamPaths  path prefixes (or suffixes) of the answer requests;
 *                           a single "*" captures every streaming response
 *                           (discovery mode for a new site)
 *   window.__lxTokenCapture true = token-capture mode: forward the Authorization
 *                           header of the page's own requests to `__lxToken`
 *                           instead of streaming bodies to `__lxEmit`
 *
 * Messages sent to Python (JSON strings):
 *   {"kind":"start","status":200,"url":"..."}  response headers received
 *   {"kind":"chunk","text":"..."}              decoded stream text (may split frames)
 *   {"kind":"end"}                             stream finished normally
 *   {"kind":"error","message":"..."}           transport failure / non-2xx status
 *
 * All SSE / JSON parsing happens in Python so a site adapter stays unit
 * testable: see app/providers/web/base.py.
 */
(() => {
  if (window.__lxDriverInstalled) return;
  window.__lxDriverInstalled = true;

  const paths = Array.isArray(window.__lxStreamPaths) ? window.__lxStreamPaths : [];
  const captureTokens = window.__lxTokenCapture === true;
  if (paths.length === 0 && !captureTokens) return;

  // Token-capture mode: report the Authorization header the page itself sends,
  // so a direct-HTTP provider can be configured without opening devtools.
  const forwardToken = (authorization, url) => {
    if (!captureTokens || !authorization) return;
    try {
      const fn = window.__lxToken;
      if (typeof fn === 'function') fn(JSON.stringify({ authorization: String(authorization), url: String(url || '') }));
    } catch (_) {
      /* binding not ready */
    }
  };

  const headerValue = (headers, name) => {
    try {
      const h = new Headers(headers || {});
      return h.get(name);
    } catch (_) {
      return null;
    }
  };

  const isStreamUrl = (url) => {
    let path;
    try {
      path = new URL(String(url), location.href).pathname;
    } catch (_) {
      return false;
    }
    if (paths.indexOf('*') !== -1) return true;
    return paths.some((p) => path === p || path.startsWith(p + '/') || path.endsWith(p));
  };

  let streamSeq = 0;

  const emit = (obj) => {
    try {
      const fn = window.__lxEmit;
      if (typeof fn === 'function') {
        // Site + account let one Python callback serve every browser context.
        obj.__site = window.__lxSite || '';
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
    if (captureTokens) {
      const auth =
        headerValue(init && init.headers, 'authorization') ||
        headerValue(input && input.headers, 'authorization');
      if (auth) forwardToken(auth, url);
    }
    const resp = await origFetch.apply(this, arguments);
    if (!isStreamUrl(url)) return resp;

    const id = ++streamSeq;
    emit({
      kind: 'start',
      id,
      status: resp.status,
      url: String(url),
      contentType: (resp.headers && resp.headers.get('content-type')) || '',
    });
    if (!resp.ok || !resp.body) {
      emit({ kind: 'error', id, message: 'HTTP ' + resp.status });
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
          if (text) emit({ kind: 'chunk', id, text });
        }
        const tail = decoder.decode();
        if (tail) emit({ kind: 'chunk', id, text: tail });
        emit({ kind: 'end', id });
      } catch (e) {
        emit({ kind: 'error', message: String((e && e.message) || e) });
      }
    })();
    return resp;
  };

  // ---- XMLHttpRequest ----------------------------------------------------
  const origOpen = XMLHttpRequest.prototype.open;
  const origSend = XMLHttpRequest.prototype.send;
  const origSetHeader = XMLHttpRequest.prototype.setRequestHeader;

  XMLHttpRequest.prototype.setRequestHeader = function (name, value) {
    if (captureTokens && String(name).toLowerCase() === 'authorization') {
      forwardToken(value, this.__lxUrl || '');
    }
    return origSetHeader.apply(this, arguments);
  };

  XMLHttpRequest.prototype.open = function (method, url) {
    this.__lxStream = isStreamUrl(url);
    this.__lxUrl = String(url || '');
    return origOpen.apply(this, arguments);
  };

  XMLHttpRequest.prototype.send = function () {
    if (this.__lxStream) {
      let offset = 0;
      let started = false;
      const id = ++streamSeq;
      const flush = () => {
        let text = '';
        try {
          text = this.responseText || '';
        } catch (_) {
          return;
        }
        if (text.length > offset) {
          emit({ kind: 'chunk', id, text: text.slice(offset) });
          offset = text.length;
        }
      };
      this.addEventListener('readystatechange', () => {
        if (this.readyState >= 2 && !started) {
          started = true;
          const type = this.getResponseHeader
            ? this.getResponseHeader('content-type') || ''
            : '';
          emit({ kind: 'start', id, status: this.status, url: this.__lxUrl || '', contentType: type });
        }
      });
      this.addEventListener('progress', flush);
      this.addEventListener('loadend', () => {
        flush();
        if (this.status >= 200 && this.status < 300) emit({ kind: 'end', id });
        else emit({ kind: 'error', id, message: 'HTTP ' + this.status });
      });
    }
    return origSend.apply(this, arguments);
  };

  // ---- WebSocket ---------------------------------------------------------
  // Several sites (MiniMax among them) stream the answer over a socket, where
  // there is no response body to read. Mirror every inbound frame instead.
  const OrigWebSocket = window.WebSocket;
  if (OrigWebSocket) {
    const WrappedWebSocket = function (url, protocols) {
      const socket = protocols === undefined
        ? new OrigWebSocket(url)
        : new OrigWebSocket(url, protocols);
      if (isStreamUrl(url)) {
        const id = ++streamSeq;
        emit({ kind: 'start', id, status: 0, url: String(url), contentType: 'websocket' });
        socket.addEventListener('message', (ev) => {
          try {
            if (typeof ev.data === 'string' && ev.data) emit({ kind: 'chunk', id, text: ev.data });
          } catch (_) {
            /* ignore */
          }
        });
        socket.addEventListener('close', () => emit({ kind: 'end', id }));
      }
      return socket;
    };
    WrappedWebSocket.prototype = OrigWebSocket.prototype;
    for (const key of ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED']) {
      WrappedWebSocket[key] = OrigWebSocket[key];
    }
    window.WebSocket = WrappedWebSocket;
  }
})();
