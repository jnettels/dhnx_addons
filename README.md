# DHNx Addons

This package contains a collection of functions useful for workflows with
[DHNx](https://github.com/oemof/DHNx), [LPagg](https://github.com/jnettels/lpagg),
and GIS-data in general in the context of municipal heat planning.

Some functions of the script are specific to Germany.

This is not a stable release and breaking changes will occur often and
without warning.

## Example workflow

This package provides a default workflow that perfoms the following:
- Take a polygon defining an area as input
- Download OpenStreetMap building and street data
- Assign a status "heated" depending on the type of each building
- Assign a random distribution of construction years
- Assign a random refurbishment status depending on the building type
  and construction year based on typical distributions from the literature
- Assign a specific heat demand based on construction year, refurbishment
  status and building type from the literature
- Estimate domestic hot water demand based on the building type
- Calculate the heated reference area based on the building ground area
  from the OpenStreetMap-Data and an estimation of the number of floors
- Apply climate correction factor based on the TRY-region
- Based on the gathered heat demand for each building, create load profiles
  for each building with LPagg
- As weather data, the old DWD TRY (2011) is used for the appropriate region
- (It is recommended to download and use the current DWD TRY (2017) data
  for your location from https://kunden.dwd.de/obt/)
- Choose a random building as a producer for a district heating grid
- Optimize the installation of a district heating grid along the
  streets with DHNx, choosing paths and required diameters for the pipes
- Simulate the heating grid to determine pressure loss, flow rate and
  temperature distribution within the network


## Installation

### TLDR
This project needs to be installed with pip, because not all dependencies
are found on conda.

Create an environment (named ``work`` in this example) with either ``venv``
```bash
python -m venv work
source work/bin/activate  # on Linux
work\Scripts\activate  # on Windows
```

or ``conda``

```bash
conda create --name=work python=3.13
conda activate work
```

then install dhnx_addons with its dependencies via ``pip``:

```bash
pip install "dhnx_addons @ https://github.com/jnettels/dhnx_addons/archive/main.tar.gz"
```
(This installs the package from this GitHub repository. ``dhnx_addons`` is not
yet published on pypi.)

### Detailed Information

- Create a dedicated python virtual environment or conda environment
  for the project
- If you want to use conda, the recommended installation is ``miniconda``
  from https://www.anaconda.com/download/success
  - On windows, if the Terminal is used with PowerShell, do not forget to run
    ``conda init powershell`` (which might require administrator rights)
- For development work:
  - Install ``git``, e.g. with ``winget install Git.Git`` if available
  - Download (clone) this repository with ``git clone https://github.com/jnettels/dhnx_addons.git``
  - Change directory into the new folder ``cd dhnx_addons``
  - Installed the package in editable mode with
    ``pip install -e .[dev]``
- If you want to use your environment in ``Spyder``, you will likely need to
  install ``spyder-kernels``. But ``Spyder`` will inform about the required version if necessary
- ``dhnx`` requires a solver to perform its optimization, e.g. the free
  ``cbc`` or ``gurobi`` (which is faster)
- The solver ``cbc`` (https://github.com/coin-or/Cbc/releases/latest)
  is installed automatically to ``~\coin-or-cbc`` by the example
  workflow, if no solver is detected. Its location is added to the system
  ``path`` only during runtime, so it might not be available in other scripts
- If the user is eligible, an academic license for ``gurobi`` can be obtained
  at https://www.gurobi.com/downloads/end-user-license-agreement-academic/
- To test the example OpenStreetMap workflow, run
  ``python examples/dhnx_example.py``
