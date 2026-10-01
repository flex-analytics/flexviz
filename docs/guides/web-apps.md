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

- Give the `Dashboard` the same data as the source: the registered frame, or a
  lazy scan of the same file. A scan reads no rows until a query runs. A
  `Dashboard()` without data gives figures without a source name, so each
  update fails and the panels stay empty.
- Give `share_url()` the name of the registered source in `source_name`.
- Use the same prefix in the mount and in `server_url`. If a proxy serves the
  web app under a path, such as `/app`, put that path in front:
  `server_url="/app/flexviz"`.
- Register all sources before the server starts.

## Streamlit

Streamlit 1.57 or newer can add routes to its own server with `st.App`. Use
two files. `app.py` runs one time, when the server starts. `page.py` is the
Streamlit script, which runs again after each widget change.

```python
# app.py
import flexviz
import polars as pl
import streamlit as st
from starlette.routing import Mount

flexviz.register_source("readings", pl.scan_parquet("readings.parquet"), cache=True)
app = st.App("page.py", routes=[Mount("/flexviz", app=flexviz.app)])
```

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

`st.iframe` cannot measure the height of the dashboard, so it uses 400 px. Set
the height yourself. A panel is 400 px by default, and the toolbar is about
45 px. While a selection is active, a 45 px bar with the active filters shows
at the bottom. The two stacked panels in the example thus need 890 px. See
[Who owns width and height](embedding.md#who-owns-width-and-height).

## Dash

Dash 4.2 or newer can run on FastAPI. Install it with the FastAPI extra:

```bash
pip install "dash[fastapi]"
```

Then mount FlexViz on `app.server`:

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
    src=url, style={"width": "100%", "height": "460px", "border": 0}
)

if __name__ == "__main__":
    app.run()
```

Start it with `python app.py`. The URL is a plain string, so a Dash callback
can return a new one to `html.Iframe.src`.

On the default Flask backend, or before Dash 4.2, `app.server` is a Flask app.
Apply the [Flask recipe](#flask-and-other-wsgi-apps) to it.

## Gradio

Gradio mounts into a FastAPI app with `gr.mount_gradio_app`. Mount FlexViz on
the same FastAPI app first:

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
        f'<iframe src="{url}" style="width: 100%; height: 460px; border: 0"></iframe>'
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
`mount_into(app, prefix="/flexviz")` adds FlexViz to it. Show the URL in the
iframe element of the framework.

### Flask and other WSGI apps

A WSGI app, such as Flask, cannot mount FlexViz directly, because FlexViz is an
ASGI app. Install [a2wsgi](https://pypi.org/project/a2wsgi/) to wrap FlexViz:

```bash
pip install a2wsgi
```

Then mount the wrapped app with the dispatcher of Werkzeug. The example leaves
out the source and the `url`. Build them as in the Dash example.

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
        f'<iframe src="{url}" style="width: 100%; height: 460px; border: 0"></iframe>'
    )
```

Start the app with `flask --app app run`.

## Limits

- The Python code of the web app cannot read the zoom or the selections in the
  dashboard. The **Share** button gives a URL with the full view.
- The server always queries the registered source. A filter on the frame of
  the `Dashboard`, such as `lf.filter(...)`, thus has no effect. To show a
  subset, register it as a separate source, and let a widget select the
  `source_name`.
- A widget can change the spec: data columns, traces, bins, or layout. To
  filter rows while you explore, select in a FlexViz figure.
- A new URL loads the dashboard again, and the view goes back to its start.

## Security

The FlexViz routes have no authentication. Each user who can open the web app
can query the registered sources through `/flexviz`. Before you deploy the web
app, put authentication in front of it. Make sure that the authentication also
covers the `/flexviz` routes.

Streamlit listens on all network interfaces by default. For local use, start
it with `streamlit run app.py --server.address 127.0.0.1`.

The `Host` check of `show()` and `flexviz serve` does not apply to a mounted
app. On a loopback address, wrap FlexViz in Starlette's `TrustedHostMiddleware`
before you mount it:

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware

guarded = TrustedHostMiddleware(flexviz.app, allowed_hosts=["localhost", "127.0.0.1"])
```

Mount `guarded` in place of `flexviz.app`, for example with
`app.mount("/flexviz", guarded)`. For Flask, wrap `guarded` in
`ASGIMiddleware`. FlexViz then refuses a request for another host name, which
blocks DNS rebinding.
