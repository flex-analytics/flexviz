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

## In an iframe

A `/view` URL is a complete page, so any web app can embed it in an iframe.
Build the URL with `share_url()` against a server that already serves the
source. The `server_url` must be reachable from the browser; `127.0.0.1` only
works when the browser and Python server run on the same machine.

```python
import polars as pl
import uvicorn
from flexviz import Dashboard, LayoutSpec, ToolbarConfig, app, register_source

lf = pl.scan_parquet("readings.parquet")
register_source("readings", lf, cache=True)

dash = Dashboard(lf, cache=True)
dash.add_figure(title="Power").add_line(x="timestamp", y="power")

url = dash.share_url(
    server_url="http://127.0.0.1:8000",
    source_name="readings",
    cols=1,             # one full-width panel instead of the half-width default
    draggable=False,    # read-only: no drag, no resize, no layout button
    layout=LayoutSpec(toolbar=ToolbarConfig(show_export=False, show_import=False)),
)
print(url)

uvicorn.run(app, host="127.0.0.1", port=8000)
```

Then embed the URL. In React:

```jsx
<iframe src={url} style={{ width: "100%", height: "600px", border: 0 }} />
```

### Who owns width and height

The parent page owns the iframe box. FlexViz owns what is inside it.

- **Width is responsive.** Panels are a 12-column grid at `width: 100%`, and
  the charts resize with their container. `cols=1` gives one full-width panel.
- **Height is fixed.** A panel spans `GridItem.h` grid rows of 80 px, so it is
  `h * 80` pixels tall. `gap` does not change this, and neither does
  `draggable`. The page is as tall as its panels plus the toolbar. It does not
  stretch to fill the iframe.

To approach a given iframe height, set `GridItem.h` yourself:

```python
from flexviz import GridItem, LayoutSpec

uid = dash.to_spec().figures[0].uid
url = dash.share_url(
    server_url="http://127.0.0.1:8000",
    source_name="readings",
    draggable=False,
    layout=LayoutSpec(grid_items=[GridItem(fig_uid=uid, x=0, y=0, w=12, h=7)]),
)
```

`h=7` is 560 px of panel. Leave room for the toolbar above it, or let the
iframe scroll. There is no mode that makes the dashboard fill its parent.

!!! warning "`height=` is notebook-only"
    `show(height=...)` sets the height of the inline notebook iframe. It does
    not size a browser page, and `/view` never reads it. For an iframe you own,
    set the height on the iframe element.

See [Customizing](customizing.md) for the full `LayoutSpec`, `GridItem` and
`ToolbarConfig` reference.
