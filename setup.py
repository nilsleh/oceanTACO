"""Setuptools hook enforcing the documented wheel boundary."""

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


_EXCLUDED_PACKAGE_PREFIXES = (
    "ocean_taco.benchmarks",
    "ocean_taco.dataset",
    "ocean_taco.generate_dataset",
    "ocean_taco.viz",
)

# The tutorial notebooks `plot_hurricane_milton` and
# `plot_hurricane_milton_cross_product` import these modules, so they must ship
# even though the rest of `ocean_taco.viz` stays repository-only. Both depend
# only on shipped modules (`catalog`, `geobox`, `retrieve`) plus matplotlib and
# cartopy, which are core dependencies. The two package `__init__` files are
# docstring-only placeholders and are needed to make the path importable.
_INCLUDED_VIZ_MODULES = frozenset(
    {
        ("ocean_taco.viz", "__init__"),
        ("ocean_taco.viz.paper", "__init__"),
        ("ocean_taco.viz.paper", "plot_hurricane_milton"),
        ("ocean_taco.viz.paper", "plot_hurricane_milton_cross_product"),
    }
)


def _is_shipped(module: tuple[str, str, str]) -> bool:
    """Return whether a (package, module, path) triple belongs in the wheel."""
    package, name, _path = module
    if (package, name) in _INCLUDED_VIZ_MODULES:
        return True
    if package.startswith(_EXCLUDED_PACKAGE_PREFIXES):
        return False
    return name != "test" and not name.startswith("test_")


class build_py(_build_py):
    """Build only modules belonging to the shipped package surface."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [module for module in modules if _is_shipped(module)]


setup(cmdclass={"build_py": build_py})
