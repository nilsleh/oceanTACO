"""Setuptools hook enforcing the documented wheel boundary."""

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


_EXCLUDED_PACKAGE_PREFIXES = (
    "ocean_taco.benchmarks",
    "ocean_taco.dataset",
    "ocean_taco.generate_dataset",
    "ocean_taco.viz",
)


class build_py(_build_py):
    """Build only modules belonging to the shipped package surface."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            module
            for module in modules
            if not module[0].startswith(_EXCLUDED_PACKAGE_PREFIXES)
            and module[1] != "test"
            and not module[1].startswith("test_")
        ]


setup(cmdclass={"build_py": build_py})
