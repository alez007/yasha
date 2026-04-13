VERSION := $(shell grep -m1 '^version' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
MAJOR   := $(shell echo $(VERSION) | cut -d. -f1)
MINOR   := $(shell echo $(VERSION) | cut -d. -f2)
PATCH   := $(shell echo $(VERSION) | cut -d. -f3)

.PHONY: test lint lint-fix release-patch release-minor release-major _release

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

lint-fix:
	uv run ruff check --fix .
	uv run ruff format .

release-patch:
	$(eval NEW_VERSION := $(MAJOR).$(MINOR).$(shell echo $$(($(PATCH)+1))))
	@$(MAKE) _release NEW_VERSION=$(NEW_VERSION)

release-minor:
	$(eval NEW_VERSION := $(MAJOR).$(shell echo $$(($(MINOR)+1))).0)
	@$(MAKE) _release NEW_VERSION=$(NEW_VERSION)

release-major:
	$(eval NEW_VERSION := $(shell echo $$(($(MAJOR)+1))).0.0)
	@$(MAKE) _release NEW_VERSION=$(NEW_VERSION)

_release:
	@if [ "$$(git branch --show-current)" != "main" ]; then echo "Error: releases must be made from the main branch" >&2; exit 1; fi
	@if [ -n "$$(git status --porcelain)" ]; then echo "Error: working tree is dirty, commit or stash changes first" >&2; exit 1; fi
	@git pull --rebase origin main
	@echo "Bumping version: $(VERSION) -> $(NEW_VERSION)"
	@sed -i '0,/^version = ".*"/{s/^version = ".*"/version = "$(NEW_VERSION)"/}' pyproject.toml
	@uv lock
	@# --- auto-update CHANGELOG.md ---
	@PREV_TAG=$$(git describe --tags --abbrev=0 2>/dev/null || echo ""); \
	if [ -n "$$PREV_TAG" ]; then \
		RANGE="$$PREV_TAG..HEAD"; \
	else \
		RANGE=""; \
	fi; \
	ADDED=$$(git log $$RANGE --pretty=format:'%s' --no-merges | grep -iE '^feat(\(.*\))?:' | sed 's/^[^:]*: */- /' || true); \
	FIXED=$$(git log $$RANGE --pretty=format:'%s' --no-merges | grep -iE '^fix(\(.*\))?:' | sed 's/^[^:]*: */- /' || true); \
	CHANGED=$$(git log $$RANGE --pretty=format:'%s' --no-merges | grep -iE '^(refactor|perf|docs|chore|build|ci|style|test)(\(.*\))?:' | sed 's/^[^:]*: */- /' || true); \
	ENTRY="## [$(NEW_VERSION)] - $$(date +%Y-%m-%d)"; \
	if [ -n "$$ADDED" ]; then ENTRY="$$ENTRY\n\n### Added\n$$ADDED"; fi; \
	if [ -n "$$FIXED" ]; then ENTRY="$$ENTRY\n\n### Fixed\n$$FIXED"; fi; \
	if [ -n "$$CHANGED" ]; then ENTRY="$$ENTRY\n\n### Changed\n$$CHANGED"; fi; \
	sed -i "/^The format is based on/a\\\\n$$ENTRY" CHANGELOG.md
	@git add pyproject.toml uv.lock CHANGELOG.md
	@git commit -m "release: v$(NEW_VERSION)"
	@git tag -a "v$(NEW_VERSION)" -m "Release v$(NEW_VERSION)"
	@git push origin main --follow-tags
	@echo "Done. GitHub Actions will build and publish the release."
