"""Setuptools hook for the release-wheel boundary.

Package discovery can exclude subpackages, but setuptools otherwise copies every
``*.py`` directly below a discovered package.  Keep repository tests out of the
wheel even when an untracked local test file is present.
"""

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Build package modules while excluding pytest-style modules."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [module for module in modules if not module[1].startswith("test_")]


setup(cmdclass={"build_py": build_py})
