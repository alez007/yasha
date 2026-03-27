VERSION := $(shell grep '^version' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
MAJOR   := $(shell echo $(VERSION) | cut -d. -f1)
MINOR   := $(shell echo $(VERSION) | cut -d. -f2)
PATCH   := $(shell echo $(VERSION) | cut -d. -f3)

.PHONY: release-patch release-minor release-major _release

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
	@echo "Bumping version: $(VERSION) -> $(NEW_VERSION)"
	@sed -i 's/^version = ".*"/version = "$(NEW_VERSION)"/' pyproject.toml
	@git add pyproject.toml
	@git commit -m "release: v$(NEW_VERSION)"
	@git tag -a "v$(NEW_VERSION)" -m "Release v$(NEW_VERSION)"
	@git push origin main --follow-tags
	@echo "Done. GitHub Actions will build and publish the release."
