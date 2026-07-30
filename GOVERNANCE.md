# Governance

Modelship is run under a BDFL (Benevolent Dictator For Life) model. This page states plainly how decisions get made, so contributors know what to expect before investing time in a PR or proposal.

## Decision authority

Alex Margarit ([@alez007](https://github.com/alez007)) has final say on everything: PR acceptance, architecture and scope, roadmap direction, and release timing. There is no voting process or RFC process — disagreements are resolved by the maintainer's judgment, not by consensus.

[CODEOWNERS](.github/CODEOWNERS) and branch protection on `main` enforce that nothing merges without maintainer review. This document is the human-readable reason why: a maintainer-approved PR reflects one person's decision, not a committee's.

## Maintainers

Currently just the one: Alex Margarit. Additional maintainers may be added at the BDFL's discretion, if and when it makes sense for the project.

## Scope guidance for contributors

Bug fixes, docs, and incremental improvements that fit the existing architecture are generally welcome without prior discussion — open a PR. Anything that changes public API surface, adds a new dependency, or shifts project direction is worth raising as an issue first, since it's more likely to need back-and-forth before it's mergeable.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the practical mechanics (setup, code style, CLA).
