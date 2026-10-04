"""The image must be reproducible from this directory alone.

The image this replaced (soapboxbuild/pdf-chart-parser, built by hand as
`:v2`) cloned upstream's default branch at build time and re-resolved every
dependency, so two builds of the same Dockerfile could ship different code. It
also fell back silently to a build without the `raster` extra if that install
failed. These checks keep both from coming back.
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
# Instructions only: a comment may describe what the build used to do.
DOCKERFILE = "\n".join(
    line for line in (HERE / "Dockerfile").read_text().splitlines()
    if not line.lstrip().startswith("#")
)
LOCK = (HERE / "requirements.lock").read_text()


def test_upstream_is_fetched_at_a_full_commit_sha_and_checked():
    match = re.search(r"^ARG UPSTREAM_SHA=([0-9a-f]+)$", DOCKERFILE, re.M)
    assert match, "Dockerfile must pin upstream with ARG UPSTREAM_SHA=<sha>"
    assert len(match.group(1)) == 40
    assert 'rev-parse HEAD)" = "$UPSTREAM_SHA"' in DOCKERFILE


def test_no_unpinned_clone():
    assert "git clone" not in DOCKERFILE


def test_no_silent_fallback_install():
    assert "||" not in DOCKERFILE


def test_dependencies_come_only_from_the_hashed_lock():
    installs = [
        line for line in DOCKERFILE.splitlines()
        if "pip install" in line and "uv==" not in line
    ]
    assert installs, "expected pip installs in the Dockerfile"
    for line in installs:
        assert "--require-hashes -r /tmp/requirements.lock" in line or "--no-deps" in line, line


def test_every_locked_package_is_pinned_with_a_hash():
    entries = re.split(r"\n(?=[A-Za-z0-9])", LOCK.strip())
    packages = [e for e in entries if not e.startswith("#")]
    assert len(packages) > 20
    for entry in packages:
        assert re.match(r"^[A-Za-z0-9_.\-\[\],]+==\S+", entry), entry.splitlines()[0]
        assert "--hash=sha256:" in entry, entry.splitlines()[0]


def test_runs_as_non_root():
    users = re.findall(r"^USER (\S+)$", DOCKERFILE, re.M)
    assert users and users[-1] not in ("root", "0")
