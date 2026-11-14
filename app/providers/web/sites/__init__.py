"""Site adapters. Import through :mod:`app.providers.web.registry`, not directly.

Each module exposes ``SITE`` (a :class:`~app.providers.web.base.WebSite`) and
``RUNTIME`` (the browser or HTTP runtime holding that site's accounts, chats and
lock). To add a site:

1. write ``sites/<name>.py`` with a parser (raw stream → ``("text", delta)``)
   and the few hooks your transport needs;
2. add it to ``SITE_LIST`` in :mod:`app.providers.web.registry`;
3. point a provider block in ``config.json`` at ``"provider_type": "<name>"``.

Sites whose protocol is plain JSON/SSE need no module at all: use
``"provider_type": "web-http"`` and declare the request and JSON paths in the
provider's ``web`` block.
"""
