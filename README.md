# pdf-chart-parser (Soapbox deployment)

This repository is the Corresponding Source, under the GNU Affero General Public
License v3.0 (section 13), of the pdf-chart-parser MCP service Soapbox runs. If you
reached that service over a network, this is the source of the version you used.

It is published automatically. Do not send changes here; they would be overwritten.

## What is here

- `upstream/` is pdf-chart-parser by its upstream author,
  <https://github.com/haoxinm/pdf-chart-parser>, at commit `a07f3be1791be5211531cf99745f57580c5c080b`, unmodified.
  It is licensed AGPL-3.0-or-later; see `upstream/LICENSE` and `upstream/README.md`.
  Only the files needed to build and test the program are included.
- `auth_app.py` is Soapbox's modification: it requires a bearer token on every MCP
  request, and advertises this repository in a `Link: rel="source"` header and at
  `GET /source`.
- `strict_tool_arguments.py` is Soapbox's modification: applied by `auth_app.py`, it
  makes every upstream tool refuse, by name, an argument its signature does not declare,
  instead of silently dropping it.
- `Dockerfile`, `requirements.in`, `requirements.lock`, `requirements.txt`,
  `pytest.ini`, `.dockerignore` and `tests/` are the build and tests of the deployed
  image. The Dockerfile fetches upstream at the same commit as `upstream/`.

Soapbox's modifications are licensed under the same terms, AGPL-3.0-or-later; see
`LICENSE`.

## Version

`SOURCE_REVISION` names the Soapbox source revision this tree was built from and the
upstream commit. The deployed service's image is built from exactly these files.

## Build

```sh
docker build -t pdf-chart-parser .
docker run -e MCP_AUTH_TOKEN=<at least 32 characters> -p 8080:8080 pdf-chart-parser
```
