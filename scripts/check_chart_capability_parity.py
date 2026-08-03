#!/usr/bin/env python3
"""CI check: every loader in modelship.deploy.capabilities.ALL_CAPABILITY_LOADERS must
also appear in the Helm chart's cuda-variant capability table (helm/modelship/templates/
_helpers.tpl's modelship.capabilityResources). The two tables are maintained by hand
in lockstep (the chart can't probe like the Python side does); this catches a new
loader added to one and forgotten in the other.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelship.deploy.capabilities import ALL_CAPABILITY_LOADERS, RESOURCE_PREFIX

_CHART_DIR = Path(__file__).resolve().parent.parent / "helm" / "modelship"

_VALUES = """
redis:
  address: cache:6379
workerGroups:
  - name: cuda
    replicas: 1
    minReplicas: 1
    maxReplicas: 1
    image:
      variant: cuda
    resources:
      requests: { cpu: "1" }
      limits: { cpu: "1" }
"""


def _rendered_cuda_resources() -> dict[str, float]:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(_VALUES)
        values_path = f.name
    out = subprocess.run(
        ["helm", "template", "modelship", str(_CHART_DIR), "-f", values_path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for doc in yaml.safe_load_all(out):
        if not doc or doc.get("kind") != "RayCluster":
            continue
        for group in doc["spec"]["workerGroupSpecs"]:
            resources = group["rayStartParams"].get("resources")
            if resources:
                # KubeRay's rayStartParams.resources quirk: the value is itself a
                # JSON-quoted, escaped JSON object (see modelship.capabilityResources).
                return json.loads(json.loads(resources))
    raise RuntimeError("no 'resources' rayStartParam found in rendered chart output")


def main() -> None:
    rendered = _rendered_cuda_resources()
    rendered_loaders = {name.removeprefix(RESOURCE_PREFIX) for name in rendered}
    missing = set(ALL_CAPABILITY_LOADERS) - rendered_loaders
    if missing:
        print(
            f"error: loader(s) {sorted(missing)} are in "
            "modelship.deploy.capabilities.ALL_CAPABILITY_LOADERS but missing from the Helm chart's "
            "cuda capability table (helm/modelship/templates/_helpers.tpl's modelship.capabilityResources). "
            "Add them there too.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"OK: {sorted(ALL_CAPABILITY_LOADERS)} all present in the chart's cuda capability table.")


if __name__ == "__main__":
    main()
