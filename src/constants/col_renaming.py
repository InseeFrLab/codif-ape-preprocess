COL_RENAMING = {
    "cj": "CJ",  # specific to sirene 4
    "activ_nat_et": "NAT",
    "liasse_type": "TYP",
    "activ_surf_et": "SRF",
    "activ_perm_et": "CRT",  # specific to sirene 4
    "activ_sec_agri_et_clean": "AGRI",  # specific to sirene 4 - textual feature
    "activ_nat_lib_et_clean": "NAT_LIB",  # specific to sirene 4 - textual feature
    "activ_sec_agri_et_cleaned": "AGRI",
    "activ_nat_lib_et_cleaned": "NAT_LIB",
}

SURFACE_COLS = ["SRF"]

TEXTUAL_FEATURES = ["AGRI", "NAT_LIB"]
CATEGORICAL_FEATURES = ["CJ", "NAT", "TYP", "SRF", "CRT"]
TEXT_FEATURE = "libelle"
