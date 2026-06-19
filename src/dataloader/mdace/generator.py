"""MDACE-ICD: MIMIC Documents Annotated with Code Evidence."""

import typing as typ

import datasets
import polars as pl

from dataloader import mimic_utils
from dataloader.constants import PROJECT_ROOT


logger = datasets.logging.get_logger(__name__)

_ANNOTATIONS_PATH = PROJECT_ROOT / "data/mdace/processed/mdace_inpatient_annotations.parquet"
_MEDICAL_CODING_SYSTEMS_DIR = PROJECT_ROOT / "data/medical-coding-systems/icd"

_CITATION = """
@inproceedings{cheng-etal-2023-mdace,
    title = "{MDACE}: {MIMIC} Documents Annotated with Code Evidence",
    author = "Cheng, Hua  and
      Jafari, Rana  and
      Russell, April  and
      Klopfer, Russell  and
      Lu, Edmond  and
      Striner, Benjamin  and
      Gormley, Matthew",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.416",
    pages = "7534--7550",
}
"""

_DESCRIPTION = """
MDACE-ICD9: A medical coding dataset created from MDACE for inpatient records with ICD-9 diagnosis and procedure codes.
It includes annotated evidence spans, supports multi-label classification tasks,
and is split into train/validation/test sets.
"""

# Consolidated, git-tracked split file (one row per note id, `split` column in
# {train,val,test}). Mirrors the MIMIC-III/IV `*_split.feather` layout so every
# dataset's splits live under data/splits/. The membership is byte-identical to
# the previous per-split MDace-ev-*.csv files (which were read with a header row,
# so each file's first line was dropped -- that exact id set is preserved here).
_SPLITS_PATH = PROJECT_ROOT / "data/splits/mdace_icd10_split.feather"


class MdaceConfig(datasets.BuilderConfig):
    """BuilderConfig for MDACE-ICD9."""

    def __init__(self, **kwargs: typ.Any):
        """BuilderConfig for MDACE-ICD9.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MdaceConfig, self).__init__(**kwargs)


class MDACE_ICD10(datasets.GeneratorBasedBuilder):
    """MDACE-ICD9: A dataset of inpatient records annotated with ICD-9 diagnosis and procedure codes."""

    BUILDER_CONFIGS = [
        MdaceConfig(
            name="icd10cm",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 diagnosis codes of 3 digits and evidence spans.",
        ),
        MdaceConfig(
            name="icd10pcs",
            version=datasets.Version("1.0.0", ""),
            description="MDACE inpatient subset dataset with ICD-10 procedure codes of 4 digits and evidence spans.",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    mimic_utils.ID_COLUMN: datasets.Value("int64"),
                    mimic_utils.ROW_ID_COLUMN: datasets.Value("string"),
                    "note_type": datasets.Value("string"),
                    "note_subtype": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "annotations": datasets.Sequence(
                        {
                            "code": datasets.Value("string"),
                            "code_type": datasets.Value("string"),
                            "code_description": datasets.Value("string"),
                            "spans": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                        }
                    ),
                    "classes": datasets.Value("string"),
                }
            ),
            citation=_CITATION,
        )

    def aggregate_rows(self, data: pl.DataFrame) -> pl.DataFrame:
        """Process the MedDec data."""
        return data.group_by(
            [
                mimic_utils.ID_COLUMN,
                mimic_utils.ROW_ID_COLUMN,
                "note_type",
                "note_subtype",
                "text",
            ]
        ).agg(
            [
                # Collect all annotations as a list of dictionaries
                pl.struct(
                    [
                        "code",
                        "code_type",
                        "spans",
                    ]
                ).alias("annotations")
            ]
        )

    def _split_generators(  # type: ignore
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        # Load the consolidated split file and recover the per-split id lists.
        split_table = pl.read_ipc(_SPLITS_PATH, memory_map=False).rename(
            {"_id": mimic_utils.ID_COLUMN}
        )
        # Ensure note_id is the same type in both `splits` and `data`.
        split_table = split_table.with_columns(pl.col(mimic_utils.ID_COLUMN).cast(pl.Int64))
        splits = {
            name: split_table.filter(pl.col("split") == name)
            for name in ("train", "val", "test")
        }

        data = self._process_data()
        aggregated_data = self.aggregate_rows(data)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data": aggregated_data.filter(
                        pl.col(mimic_utils.ID_COLUMN).is_in(splits["train"][mimic_utils.ID_COLUMN])
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data": aggregated_data.filter(
                        pl.col(mimic_utils.ID_COLUMN).is_in(splits["val"][mimic_utils.ID_COLUMN])
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data": aggregated_data.filter(
                        pl.col(mimic_utils.ID_COLUMN).is_in(splits["test"][mimic_utils.ID_COLUMN])
                    )
                },
            ),
        ]

    def _process_data(
        self,
    ) -> pl.DataFrame:
        """Processes the raw dataset."""
        mdace = pl.read_parquet(_ANNOTATIONS_PATH)

        # The processed parquet stores the admission id as `_id`; the rest of the
        # generator (splits join, group_by, features) expects `hadm_id`.
        if "_id" in mdace.columns and mimic_utils.ID_COLUMN not in mdace.columns:
            mdace = mdace.rename({"_id": mimic_utils.ID_COLUMN})

        if "icd9" in self.config.name:
            mdace = mdace.filter(
                pl.col("diagnosis_code_type").str.contains("icd9") | pl.col("diagnosis_code_type").str.contains("icd9")
            )
        else:
            mdace = mdace.filter(pl.col("code_type").str.contains("icd10"))

        if "cm" in self.config.name:
            mdace = mdace.filter(pl.col("code_type").str.contains("cm"))
        elif "pcs" in self.config.name:
            mdace = mdace.filter(pl.col("code_type").str.contains("pcs"))

        # Transform spans to list[list[int]] format. The processed parquet may
        # already store spans as List(List(Int64)) (some prepare steps emit the
        # final format directly); only convert when they are still in the
        # {start, end} struct form, otherwise the map_elements below would fail
        # trying to index an int list with the "start"/"end" string keys.
        spans_dtype = mdace.schema["spans"]
        inner = spans_dtype.inner if isinstance(spans_dtype, pl.List) else None
        if isinstance(inner, pl.Struct):
            mdace = mdace.with_columns(
                pl.col("spans")
                .map_elements(
                    lambda span_list: [[span["start"], span["end"]] for span in span_list],
                    return_dtype=pl.List(pl.List(pl.Int64)),
                )
                .alias("spans")
            )

        return mdace

    def _generate_code_mapping(self, old_codes: set[str], new_codes: set[str]) -> dict[str, str]:
        """Generate a mapping from old ICD-9 codes to new ICD-10 codes."""
        mapping = {}
        for new_code in new_codes:
            temp_code = new_code
            while temp_code and temp_code not in old_codes:
                temp_code = temp_code[:-1]
            mapping[new_code] = temp_code if temp_code else new_code
        return mapping

    def _generate_examples(  # type: ignore
        self, data: pl.DataFrame
    ) -> typ.Generator[tuple[int, dict[str, typ.Any]], None, None]:  # type: ignore
        """Generate examples from the dataset."""
        for row in data.to_dicts():
            yield row[mimic_utils.ROW_ID_COLUMN], row
