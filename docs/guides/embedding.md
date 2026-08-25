# Embedding

`show()` is a convenience for local exploration: it registers the data
source, starts a server thread, and opens a browser. For anything longer
lived, run the server yourself.

## Standalone server

Register named sources, then run the FastAPI app like any other:

```python
import polars as pl
import uvicorn
from flexviz import app, register_source

register_source("trips", pl.scan_parquet("trips.parquet"))
register_source("weather", pl.scan_parquet("weather.parquet"), cache=True)

uvicorn.run(app, host="127.0.0.1", port=8000)
```

Register all sources before the server starts; the registry is the only
server-side state and is read-only during request handling. Pass
`cache=True` for static sources to enable
[caching and live brushing](caching-and-live-brushing.md).

Because the server is stateless, it scales horizontally without session
affinity: any replica can answer any request.

## Mounting into an existing FastAPI app

`mount_into` adds the FlexViz routes to an app you already have:

```python
from fastapi import FastAPI
from flexviz import mount_into, register_source

app = FastAPI()
register_source("trips", lf)
mount_into(app, prefix="/flexviz")
```

The FlexViz endpoints (`/update`, `/dashboard/update`, `/share`, `/view`,
`/sources`) then live under the prefix. The mounted app brings its own gzip
middleware, so responses are compressed regardless of the host app's setup.
For Flask or other WSGI hosts, use
`werkzeug.middleware.dispatcher.DispatcherMiddleware` instead.

## Serving a dashboard from a URL

A browser opens a view through `GET /view?spec=<encoded>`. To hand out a
dashboard URL from your own code, build the spec and ask the server to encode
it, or use the toolbar's share button from a rendered view (see
[Sharing views](sharing.md)). The `/view` page talks to the API with
page-relative URLs, so it works unchanged behind a reverse proxy or path
prefix.
