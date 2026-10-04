# Web apps

You can show a FlexViz dashboard in a Streamlit, Dash, Gradio, or other Python
web app. The web app owns the page, the widgets, and the layout. FlexViz owns
the figures and runs their queries on your data. To run FlexViz as a separate
server instead, see [Embedding](embedding.md).

The procedure is the same for each web framework:

1. Register the data source with `register_source()`.
2. Mount the FlexViz app on the server of the web app, under `/flexviz`.
3. Build a `/view` URL with `share_url(server_url="/flexviz", source_name=...)`.
4. Show the URL in an iframe.

The web app and FlexViz then use one server and one port, so you deploy one
process. The URL has no host name, so it works on each host that serves the web
app.

These rules apply to every web framework:

- Give `share_url()` the name of the registered source in `source_name`.
- Use the same prefix in the mount and in `server_url`.
- If a proxy serves the web app under a path, such as `/app`, put that path in
  front: `server_url="/app/flexviz"`.
- Register all sources before the server starts.
- Do not change the data of a registered source while the server runs. To show
  new data, restart the server.

A broken rule shows only in the browser. The `Dashboard` does not need the
data: `Dashboard()` works, because `share_url()` puts `source_name` in each
figure. If the `Dashboard` has data, FlexViz reads its schema only to check
linked axes. A lazy scan reads no rows until a query runs, so the web app can
build its `Dashboard` on a scan at no cost. After a change of a scanned file,
the updates fail. With `cache=True`, the first view of each figure can show the
old data. See [Caching and live brushing](caching-and-live-brushing.md).

## Example data

The examples read `readings.parquet`, a file with a `timestamp` and a `power`
column.

To make a test file with 2 million rows, run this script one time:

```python
# make_data.py
import numpy as np
import polars as pl

n = 2_000_000
pl.select(
    timestamp=pl.datetime(2026, 1, 1) + pl.duration(seconds=pl.int_range(n)),
    power=pl.Series(np.random.default_rng(0).standard_normal(n).cumsum()),
).write_parquet("readings.parquet")
```

## Streamlit

Streamlit 1.57 or newer can add routes to its own server with `st.App`. The
app then has two files. `app.py` runs one time, when the server starts.
`page.py` is the Streamlit script, which runs again after each widget change.

Put the source and the mount in `app.py`:

```python
# app.py
import flexviz
import polars as pl
import streamlit as st
from starlette.routing import Mount

flexviz.register_source("readings", pl.scan_parquet("readings.parquet"), cache=True)
app = st.App("page.py", routes=[Mount("/flexviz", app=flexviz.app)])
```

Put the page in `page.py`:

```python
# page.py
import polars as pl
import streamlit as st
from flexviz import Dashboard

st.title("Sensor readings")
bins = st.slider("Histogram bins", 10, 200, 60)


@st.cache_data
def view_url(bins: int) -> str:
    dashboard = Dashboard(pl.scan_parquet("readings.parquet"), cache=True)
    dashboard.add_figure(title="Power").add_line(x="timestamp", y="power")
    dashboard.add_figure(title="Distribution").add_histogram(x="power", bins=bins)
    return dashboard.share_url(server_url="/flexviz", source_name="readings", cols=1)


st.iframe(view_url(bins), height=900)
```

Start the app:

```bash
streamlit run app.py
```

Streamlit finds the `app` object in `app.py`. It then serves the page and
FlexViz on one port, 8501 by default.

### Keep the view across reruns

A new `Dashboard` gets new uids, so each run of `page.py` gives a new URL. The
iframe then loads the dashboard again, and the user loses the zoom and the
selections.

`st.cache_data` prevents this reset. It returns the same URL for the same
arguments, and Streamlit keeps an iframe whose URL does not change. In the
example, only a change of `bins` loads the dashboard again.

### Set the height

`st.iframe` cannot measure the height of the dashboard, so it uses 400 px. A
panel is 400 px by default, and the toolbar is about 45 px. While a selection
is active, a 45 px bar with the active filters shows at the bottom. The two
stacked panels in the example thus need 890 px. See
[Who owns width and height](embedding.md#who-owns-width-and-height).

Set the height of the iframe yourself, as `height=900` does in `page.py`.

## Dash

Dash 4.2 or newer can run on FastAPI.

Install Dash with the FastAPI extra:

```bash
pip install "dash[fastapi]"
```

Mount FlexViz on `app.server`:

```python
# app.py
import polars as pl
from dash import Dash, html
from flexviz import Dashboard, mount_into, register_source

lf = pl.scan_parquet("readings.parquet")
register_source("readings", lf, cache=True)

dashboard = Dashboard(lf, cache=True)
dashboard.add_figure(title="Power").add_line(x="timestamp", y="power")
url = dashboard.share_url(server_url="/flexviz", source_name="readings", cols=1)

app = Dash(__name__, backend="fastapi")
mount_into(app.server, prefix="/flexviz")
app.layout = html.Iframe(
    src=url,
    title="Sensor readings dashboard",
    style={"width": "100%", "height": "460px", "border": 0},
)

if __name__ == "__main__":
    app.run()
```

Start the app with `python app.py`.

The URL is a plain string, so a Dash callback can return a new one to
`html.Iframe.src`.

On the default Flask backend, or before Dash 4.2, apply the
[Flask recipe](#flask-and-other-wsgi-apps) to `app.server`.

## Gradio

Gradio mounts into a FastAPI app with `gr.mount_gradio_app`.

Mount FlexViz on the same FastAPI app, before Gradio:

```python
# app.py
import gradio as gr
import polars as pl
import uvicorn
from fastapi import FastAPI
from flexviz import Dashboard, mount_into, register_source

lf = pl.scan_parquet("readings.parquet")
register_source("readings", lf, cache=True)

dashboard = Dashboard(lf, cache=True)
dashboard.add_figure(title="Power").add_line(x="timestamp", y="power")
url = dashboard.share_url(server_url="/flexviz", source_name="readings", cols=1)

with gr.Blocks() as demo:
    gr.HTML(
        f'<iframe src="{url}" title="Sensor readings dashboard" '
        'style="width: 100%; height: 460px; border: 0"></iframe>'
    )

app = FastAPI()
mount_into(app, prefix="/flexviz")
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
```

!!! warning "Mount FlexViz before Gradio"
    Gradio at `path="/"` answers every path that no earlier route answers. If
    you mount Gradio first, `/flexviz/view` returns 404.

## Other web frameworks

### FastAPI and Starlette apps

A web framework that runs on FastAPI or Starlette takes the same mount. For
example, the `app` of NiceGUI is a FastAPI app, so
`mount_into(app, prefix="/flexviz")` adds FlexViz to it.

### Flask and other WSGI apps

A WSGI app, such as Flask, cannot mount FlexViz directly, because FlexViz is an
ASGI app. The [a2wsgi](https://pypi.org/project/a2wsgi/) package wraps FlexViz
as a WSGI app. The example leaves out the source and the `url`, which are the
same as in the Dash example.

Install a2wsgi:

```bash
pip install a2wsgi
```

Mount the wrapped app with the dispatcher of Werkzeug:

```python
import flexviz
from a2wsgi import ASGIMiddleware
from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)
app.wsgi_app = DispatcherMiddleware(
    app.wsgi_app, {"/flexviz": ASGIMiddleware(flexviz.app)}
)


@app.get("/")
def index():
    return (
        f'<iframe src="{url}" title="Sensor readings dashboard" '
        'style="width: 100%; height: 460px; border: 0"></iframe>'
    )
```

Start the app with `flask --app app run`.

## Limits

- The Python code of the web app cannot read the zoom or the selections in the
  dashboard. The **Share** button gives a URL with the full view.
- The server always queries the registered source. A filter on the frame of
  the `Dashboard`, such as `lf.filter(...)`, thus has no effect.
- A subset of the data needs its own registered source. A widget can then
  select it through `source_name`.
- A widget can change the spec: data columns, traces, bins, or layout. A
  selection in a FlexViz figure filters the rows of the other figures.
- A new URL loads the dashboard again, and the view goes back to its start.

## Security

The FlexViz routes have no authentication. Each user who can open the web app
can query the registered sources through `/flexviz`. The mount also serves
`/flexviz/h/N`, which reads the agent history file `.flexviz/history.jsonl` in
the working folder of the server. See the
[safety notes for agents](ai-agents.md#safety-notes).

Streamlit listens on all network interfaces by default. The `Host` check of
`show()` and `flexviz serve` does not apply to a mounted app.

Before you deploy the web app:

- Put authentication in front of it. Make sure that the authentication also
  covers the `/flexviz` routes.
- Do not run it from a folder that holds `.flexviz/history.jsonl`.

For local use:

- Start Streamlit with `streamlit run app.py --server.address 127.0.0.1`.
- Wrap FlexViz in Starlette's `TrustedHostMiddleware`, as the example that
  follows shows.

```python
import flexviz
from starlette.middleware.trustedhost import TrustedHostMiddleware

guarded = TrustedHostMiddleware(flexviz.app, allowed_hosts=["localhost", "127.0.0.1"])
```

Mount `guarded` in place of `flexviz.app`:

| Web framework | Mount                                                      |
| ------------- | ---------------------------------------------------------- |
| Streamlit     | `Mount("/flexviz", app=guarded)` in the routes of `st.App` |
| Dash          | `app.server.mount("/flexviz", guarded)`                    |
| Gradio        | `app.mount("/flexviz", guarded)`, before Gradio            |
| Flask         | `ASGIMiddleware(guarded)` in the `DispatcherMiddleware`    |

The middleware refuses a request for another host name, which blocks DNS
rebinding.
