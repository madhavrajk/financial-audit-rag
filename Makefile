.PHONY: setup index index-offline run cli check test eval retrieval docker check-secrets

setup:            ## install deps
	uv sync

index:            ## (re)build the page index with embeddings (incremental; needs EMBED_MODEL)
	uv run python -m app.ingest

index-offline:    ## BM25-only index, no API calls
	uv run python -m app.ingest --no-embed

run:              ## API + UI on http://localhost:8000
	uv run python -m app.api.server

cli:              ## interactive terminal chat
	uv run python -m app.cli

check:            ## lint, types, tests, dependency audit
	uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q && uv run pip-audit --progress-spinner off

test:
	uv run pytest -q

eval:             ## the 16 eval questions, 1 run
	uv run python -m eval.run_eval

retrieval:        ## retrieval ablation (BM25 / dense / hybrid)
	uv run python -m eval.retrieval

docker:
	docker compose up --build

check-secrets:    ## fail if an API key appears outside .env
	@if grep -rIlE '(^|[^A-Za-z0-9_-])(sk-(proj-|or-v1-)?|fw_|gsk_|sk-ant-|AIza)[A-Za-z0-9_-]{16,}' --exclude=.env --exclude-dir=.venv --exclude-dir=logs --exclude-dir=.git --exclude-dir=financebench . ; then \
	  echo "ERROR: API key found in the files above; remove it before committing."; exit 1; fi
	@echo "no keys found"
