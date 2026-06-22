"""Entry points for dataset generation commands."""


def download_main():
    """Run the raw-data download CLI."""
    from .download import main

    return main()


def format_main():
    """Run the formatting CLI."""
    from .format import main

    return main()


def build_taco_main():
    """Run the final TACO build CLI."""
    from .build_taco import main

    return main()


__all__ = ["download_main", "format_main", "build_taco_main"]
