

import pathlib
import subprocess

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

here = pathlib.Path(__file__).parent


def _clean_head(repo: pathlib.Path):
    """The commit *repo* is checked out at, or None if it is not a clean git checkout.

    A build from a dirty tree is not that commit, so it gets no stamp rather than a
    wrong one.
    """
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30, check=True
        ).stdout.strip()

    try:
        sha = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except Exception:
        return None
    return sha if len(sha) == 40 and not dirty else None


class build_py_with_commit(build_py):
    """Write the commit being built into the built package, as SWEET_python/_build_sha.py.

    pip installs SWEET from git (`pip install git+https://...@sweet/2026.10.0`) by
    cloning it and building from that checkout, so the commit is known at build time and
    nowhere afterwards: the installed copy has no .git, and only an app that sets
    SWEET_GIT_SHA could say which code it got. The stamp goes into the BUILD directory,
    never the source tree, so an editable install of a checkout can't pick up a stale one.
    """

    def run(self):
        super().run()
        sha = _clean_head(here)
        if sha and not self.dry_run:
            target = pathlib.Path(self.build_lib) / "SWEET_python" / "_build_sha.py"
            target.write_text(
                '"""Written by setup.py when this copy was built. Not in git."""\n'
                f'GIT_SHA = "{sha}"\n'
            )

# Read the abstract (unpinned) dependencies from requirements.in so consumers are
# not over-constrained. The pinned lockfile in requirements.txt is generated from
# this file with pip-tools (pip-compile) and is used for reproducible installs and
# Dependabot scanning, not for install_requires.
requirements = [
    line.strip() for line in (here / "requirements.in").read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="SWEET_python",
    # Decoration, not a version. SWEET's releases are git tags named sweet/YYYY.MM.N,
    # cut from main by .github/workflows/release.yml (see the README's Releases
    # section). This string does not follow them and is never bumped -- pip reports
    # 0.1 whichever release it installed. What a caller should read is
    # SWEET_python.__version__, which is the commit this copy was installed from; a
    # release tag names exactly one commit, so the commit identifies the release too.
    # See the package docstring.
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    cmdclass={"build_py": build_py_with_commit},
)
