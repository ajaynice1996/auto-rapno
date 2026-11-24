auto-rapno/
│
├── nnUNet/
│   └── nnUNet_raw/
│       └── Dataset202_BraTS/
│           ├── imagesTr_all/
│           └── labelsTr_all/
│
└── brain_tumor_reporting/
    │
    ├── config/
    │   ├── dataset_config.py
    │   └── label_mapping.py
    │
    ├── data/
    │
    ├── preprocessing/
    │   ├── 01_read_nifti.py
    │   ├── 02_validate_images.py
    │   └── 03_align_modalities.py
    │
    ├── segmentation/
    │   ├── 04_segmentation_analysis.py
    │   ├── 05_quantitative_metrics.py
    │   └── 06_morphological_features.py
    │
    ├── parcellation/
    │   ├── 07_register_atlas.py
    │   └── 08_regional_overlap.py
    │
    ├── visualization/
    │   ├── 09_generate_slices.py
    │   ├── 10_generate_overlays.py
    │   ├── 11_generate_3d.py
    │   └── 12_generate_contact_sheets.py
    │
    ├── reporting/
    │   ├── 13_create_findings_table.py
    │   ├── 14_prepare_llm_input.py
    │   ├── 15_generate_findings.py
    │   ├── 16_generate_impression.py
    │   └── 17_compile_report.py
    │
    ├── evaluation/
    │   ├── 18_reference_report_parser.py
    │   ├── 19_compare_reports.py
    │   └── 20_evaluation_metrics.py
    │
    ├── outputs/
    │
    ├── run_pipeline.py
    └── requirements.txt