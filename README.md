DHNx Addons
===========

This package contains a collection of functions useful for workflows with
[DHNx](https://github.com/oemof/DHNx), [LPagg](https://github.com/jnettels/lpagg),
and GIS-data in general in the context of municipal heat planning.

Some functions of the script are specific to Germany.

This is not a stable release and breaking changes will occur often and
without warning.

Example workflow
----------------

This package provides a default workflow that perfoms the following:
- Take a polygon defining an area as input
- Download OpenStreetMap building and street data
- Assign a status "heated" depending on the type of each building
- Assign a random distribution of construction years
- Assign a random refurbishment status depending on the building type
  and construction year based on typical distributions from the literature.
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
  streets with DHNx


Installation
------------

- Download and install Anaconda. On windows, if ``winget`` is installed,
  the fastest way is
  ``winget install anaconda3``
- On windows, if the Terminal is used with PowerShell, do not forget to run
  ``conda init powershell`` (which might require administrator rights)
- Download and install ``git`` to clone this repository, e.g. with
  ``winget install Git.Git``
- It is recommended to install the dependencies into a dedicated python
  environment. This repository contains an example environment file
  that can be used like this:
  ``conda env create --file environment.yaml``
- Activate the new environment with
  ``conda activate work``
- If you want to use your new environment in ``Spyder``, you will likely
  need to install ``spyder-kernels``. But ``Spyder`` will inform about the
  required version if necessary
- Some dependencies (namely ``dhnx`` and ``oemof``) can only be obtained
  via ``pip``, not ``conda``
- As of June 2023, it is recommended to install ``dhnx 0.0.3`` directly from
  ``pip``. Perform a ``dry-run`` first to see if any packages other than
  dhnx and oemof would be installed. Try to install those with conda, if
  possible. (Mixing ``pip`` and ``conda`` installations in generally not
  recommended, because ``conda`` does not keep track of the ``pip``
  installations properly.) Use
  ``pip install dhnx==0.0.3 --dry-run``
- If everything seems fine, perform the actual installation with
  ``pip install dhnx==0.0.3``
- Development on ``dhnx`` is ongoing. Newer versions might require updates
  to the current workflow. These can be tested by installing directly
  from the repository at https://github.com/oemof/DHNx
- ``dhnx`` requires a solver to perform its optimization, e.g. the free
  ``cbc`` or ``gurobi`` (which is faster)
- The solver ``cbc`` (https://github.com/coin-or/Cbc/releases/latest)
  is installed automatically to ``~\coin-or-cbc`` by the example
  workflow, if no solver is detected. Its location is added to the system
  ``path`` only during runtime, so it might not be available in other scripts
- If the user is eligible, an academic license for ``gurobi`` can be obtained
  at https://www.gurobi.com/downloads/end-user-license-agreement-academic/
- After these steps, run the following to test the example
  OpenStreetMap workflow
  ``python dhnx_addons.py``