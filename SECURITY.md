# Security Policy

## Supported versions

FlexViz is pre-1.0. Only the latest release receives security fixes.

## Reporting a vulnerability

Please report vulnerabilities privately via GitHub's
[private vulnerability reporting](https://github.com/flex-analytics/flexviz/security/advisories/new)
("Report a vulnerability" under the Security tab). Do not open a public issue
for security problems.

FlexViz is maintained part-time; we aim to acknowledge reports within 7 days.

## Deployment posture

The FlexViz server is designed for local use and trusted networks: it ships
**no authentication or authorization**. `Figure.show()` binds a local dev
server; anyone who can reach the port can query the registered data sources.
If you expose FlexViz publicly, put it behind your own auth (reverse proxy,
`mount_into()` inside an authenticated FastAPI app, or a tunnel with access
control). Reports assuming an unauthenticated public deployment of the dev
server are out of scope; bugs that let a request escape the registered data
sources or execute arbitrary code are very much in scope.
