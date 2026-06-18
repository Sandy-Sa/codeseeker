import pathlib

# The Tanner Health datasets live in the private `ml-datasets` package (corti),
# which is an optional dependency: it was removed from the project deps and is
# absent on machines that only run the public datasets (e.g. the MDACE slice).
# Import it lazily so that `import dataloader` keeps working without it; the
# `tanner-health-*` DATASET_CONFIGS entries only fail if actually requested.
try:
    from ml_datasets.tanner_health import (
        tanner_health_inpatient,
        tanner_health_surgery,
        tanner_health_ed,
    )

    TANNER_INPATIENT_PATH = str(pathlib.Path(tanner_health_inpatient.__file__))
    TANNER_SURGERY_PATH = str(pathlib.Path(tanner_health_surgery.__file__))
    TANNER_ED_PATH = str(pathlib.Path(tanner_health_ed.__file__))
except ModuleNotFoundError:
    TANNER_INPATIENT_PATH = None
    TANNER_SURGERY_PATH = None
    TANNER_ED_PATH = None
