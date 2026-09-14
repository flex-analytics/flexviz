# Layout

Dashboard layout is described by a `LayoutSpec`. Pass one to
`Dashboard.show()` or `Dashboard.share_url()` to control the grid, the gap,
and which toolbar buttons appear. See
[Customizing](../guides/customizing.md) and
[Embedding](../guides/embedding.md) for worked examples.

These models live in `flexviz.spec`:

```python
from flexviz.spec import GridItem, LayoutSpec, ToolbarConfig
```

::: flexviz.spec.LayoutSpec

::: flexviz.spec.GridItem

::: flexviz.spec.ToolbarConfig
