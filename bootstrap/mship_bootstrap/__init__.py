"""Two-stage installer for modelship.

Stage one (this package) runs on any Python >= 3.10 and owns everything that must
happen before the engine exists: variant selection, hardware validation, and
provisioning a CPython 3.12.10 environment from hash-pinned dependency lists.
Stage two is `mship-engine`, exec'd inside that environment.
"""

# Lockstep with mship-engine; `make _release` bumps both pyprojects.
__version__ = "0.7.11"
