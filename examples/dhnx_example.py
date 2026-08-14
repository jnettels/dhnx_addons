# -*- coding: utf-8 -*-
"""Call the example main method of the dhnx_addons package.
"""
import dhnx_addons


def main():
    """Run an example main method."""
    dhnx_addons.setup()

    dhnx_addons.workflow_example_openstreetmap(show_plot=True)
    # dhnx_addons.workflow_example_openstreetmap(show_plot=False)


if __name__ == '__main__':
    main()
