# RESEARCH-DRAFT BRAIN MRI REPORT

**Subject ID:** BraTSPED_00002  
**Report Status:** Automated Research Draft (Requires Qualified Radiologist Review)

---

## 1. Examination and Available Data

* **Demographic Data:** Not available from the supplied data (Age: null, Sex: null).
* **MRI Sequences Provided:** 
  * T1-weighted native (T1n)
  * T1-weighted contrast-enhanced (T1c)
  * T2-weighted FLAIR (T2f)
  * T2-weighted (T2w)
* **Image Grid & Alignment:** All sequence images, anatomical segmentations, and tumor masks are aligned to the T1n reference grid with dimensions $240 \times 240 \times 155$, isotropic voxel resolution of $1.0 \times 1.0 \times 1.0\text{ mm}^3$ ($1.0\text{ mm}^3$ per voxel), and orientation ($\text{L}, \text{P}, \text{S}$).
* **Computational Data Supplied:** Automated tumor segmentation labels (ET, NET, CC, ED), composite regions (TC, WT), SynthSeg automated anatomical segmentation labels, spatial overlap tables, and SynthSeg regional quality control (QC) scores.

---

## 2. Tumor Segmentation Summary

The computational segmentation identifies a total Whole Tumor (WT) volume of **17.497 mL** (17,497 voxels). The component breakdown is as follows:

* **Enhancing Tumor (ET, Label 1):** 3.316 mL (3,316 voxels)
* **Non-Enhancing Tumor (NET, Label 2):** 14.164 mL (14,164 voxels)
* **Cavity / Cystic Component (CC, Label 3):** 0.000 mL (0 voxels)
* **Peritumoral Edema (ED, Label 4):** 0.017 mL (17 voxels)
* **Tumor Core (TC, Composite Labels 1+2+3):** 17.480 mL (17,480 voxels)
* **Whole Tumor (WT, Composite Labels 1+2+3+4):** 17.497 mL (17,497 voxels)

*Note: TC and WT represent composite union volumes and must not be summed with individual subcomponent volumes.*

---

## 3. MRI Findings

### Direct Image Observations
* **Location & Morphology:** A well-circumscribed, round-to-oval mass is centered in the left deep gray matter / left thalamic region. The lesion exhibits distinct margins relative to surrounding brain tissue on T2-weighted images.
* **Signal Characteristics:**
  * **T1n:** The mass demonstrates predominantly hypointense signal intensity relative to adjacent brain parenchyma.
  * **T1c:** Internal heterogeneous signal is present, featuring central regions of patchy hyperintensity/enhancement surrounded by a non-enhancing peripheral zone.
  * **T2w & T2f:** The lesion exhibits hyperintense signal throughout its expanse. There is no appreciable extending T2/FLAIR hyperintensity in the surrounding cerebral parenchyma.
* **Mass Effect:** The mass expands the left thalamus, compresses the adjacent third ventricle with rightward midline shift, and impinges upon the left lateral ventricle wall, accompanied by mild secondary upstream enlargement of the lateral ventricles.

### Visual Correlation with Segmentation Mask
* **WT Overlay:** Accurately outlines the outer margin of the mass as visualized on T2w, T2f, and T1n images.
* **NET Overlay:** Maps to the non-enhancing peripheral/solid matrix of the mass.
* **ET Overlay:** Maps centrally within the core of the lesion where internal T1c hyperintensity is observed.
* **ED Overlay:** The 0.017 mL (17 voxels) label is confined to a tiny isolated point near the superolateral ventricular border on axial slice 99, lacking corresponding parenchymal T2/FLAIR signal alteration on the original MRI images.

---

## 4. Anatomical Distribution

Anatomical distribution metrics quantify voxel-wise spatial overlap between the tumor segmentation masks and automated anatomical labels. Overlap does not establish histological tissue invasion or destruction.

### Whole Tumor (WT) Top Overlaps
* **Left Thalamus:** 11.567 mL overlap (66.11% of total WT volume; occupies 95.81% of labeled left thalamus volume).
* **Left Cerebral White Matter:** 2.488 mL overlap (14.22% of total WT volume; occupies 1.54% of labeled white matter volume).
* **Left Ventral DC:** 1.694 mL overlap (9.68% of total WT volume; occupies 35.87% of labeled ventral DC volume).
* **Brain-stem:** 0.962 mL overlap (5.50% of total WT volume; occupies 5.55% of labeled brainstem volume).
* **3rd Ventricle:** 0.345 mL overlap (1.97% of total WT volume; occupies 21.38% of labeled 3rd ventricle volume, reflecting compression/displacement).

### Subcomponent Distribution Highlights
* **ET Label (3.316 mL total):** Predominantly overlaps the **left thalamus** (3.265 mL; 98.46% of ET), with minor overlap in left ventral DC (0.028 mL), left cerebral white matter (0.010 mL), 3rd ventricle (0.008 mL), and brain-stem (0.004 mL).
* **NET Label (14.164 mL total):** Mainly overlaps the **left thalamus** (8.299 mL; 58.59% of NET), **left cerebral white matter** (2.467 mL; 17.42% of NET), **left ventral DC** (1.666 mL; 11.76% of NET), **brain-stem** (0.958 mL; 6.76% of NET), and **3rd ventricle** (0.337 mL; 2.38% of NET).
* **ED Label (0.017 mL total):** Overlaps left cerebral white matter (0.011 mL), left thalamus (0.003 mL), and left caudate (0.003 mL).

---

## 5. Quantitative Summary

### Table 1: Tumor Component Volumes
| Component | Label Code | Volume (mL) | Volume ($\text{mm}^3$) | Voxel Count |
| :--- | :--- | :--- | :--- | :--- |
| **Enhancing Tumor (ET)** | 1 | 3.316 | 3316.0 | 3316 |
| **Non-Enhancing Tumor (NET)** | 2 | 14.164 | 14164.0 | 14164 |
| **Cavity / Cystic Component (CC)** | 3 | 0.000 | 0.0 | 0 |
| **Peritumoral Edema (ED)** | 4 | 0.017 | 17.0 | 17 |
| **Tumor Core (TC)** | 1, 2, 3 | 17.480 | 17480.0 | 17480 |
| **Whole Tumor (WT)** | 1, 2, 3, 4 | 17.497 | 17497.0 | 17497 |

### Table 2: Top Anatomical Overlaps by Tumor Component
| Component | Anatomical Structure | Overlap Volume (mL) | % of Component in Anatomy | % of Anatomy Occupied |
| :--- | :--- | :--- | :--- | :--- |
| **ET** | Left Thalamus | 3.265 | 98.46% | 27.04% |
| **ET** | Left Ventral DC | 0.028 | 0.84% | 0.59% |
| **ET** | Left Cerebral White Matter | 0.010 | 0.30% | 0.01% |
| **NET** | Left Thalamus | 8.299 | 58.59% | 68.74% |
| **NET** | Left Cerebral White Matter | 2.467 | 17.42% | 1.53% |
| **NET** | Left Ventral DC | 1.666 | 11.76% | 35.27% |
| **NET** | Brain-stem | 0.958 | 6.76% | 5.53% |
| **TC** | Left Thalamus | 11.564 | 66.16% | 95.78% |
| **TC** | Left Cerebral White Matter | 2.477 | 14.17% | 1.54% |
| **TC** | Left Ventral DC | 1.694 | 9.69% | 35.87% |
| **WT** | Left Thalamus | 11.567 | 66.11% | 95.81% |
| **WT** | Left Cerebral White Matter | 2.488 | 14.22% | 1.54% |
| **WT** | Left Ventral DC | 1.694 | 9.68% | 35.87% |
| **WT** | Brain-stem | 0.962 | 5.50% | 5.55% |
| **WT** | 3rd Ventricle | 0.345 | 1.97% | 21.38% |

---

## 6. Limitations

1. **Clinical & Histopathology Data:** Patient age, sex, clinical presentation, prior treatment history, and pathological diagnoses are not available from the supplied data. Tumor grade or histopathological diagnosis cannot be determined from computational segmentations.
2. **Segmentation Artifact:** The small ED volume of 0.017 mL (17 voxels) lacks corresponding peritumoral hyperintensity on T2/FLAIR images and likely represents minor automated segmentation threshold noise at tissue boundaries.
3. **Anatomical Overlap vs. Pathological Invasion:** Spatial overlap metrics reflect computational intersection of voxel grids. Mass effect, distortion of the third ventricle, and registration software limits influence these metrics and must not be interpreted as histological invasion.
4. **Quality Control Scores:** SynthSeg regional QC scores (e.g., general grey matter: 0.6801, general white matter: 0.7837, thalamus: 0.8725, brainstem: 0.8672) evaluate automated atlas segmentation performance and do not represent clinical diagnostic accuracy or tumor confidence scores.

---

## 7. Research-Draft Impression

1. **Left Deep Gray Matter / Thalamic Mass:** Centered in the left thalamic region with a total Whole Tumor (WT) volume of 17.497 mL, comprised predominantly of non-enhancing tumor (NET: 14.164 mL) with a central enhancing tumor component (ET: 3.316 mL). Cystic components are absent (CC: 0.000 mL).
2. **Anatomical Correspondence:** Spatial overlap is overwhelmingly localized to the left thalamus (11.567 mL overlap, occupying 95.81% of the labeled left thalamus volume), with additional minor spatial overlap in the left ventral DC, left cerebral white matter, brainstem, and third ventricle.
3. **Mass Effect:** Local mass effect results in rightward displacement/compression of the third ventricle and displacement of the left lateral ventricle, with secondary mild ventricular enlargement.
4. **Absence of Significant Edema:** Peritumoral edema is negligible (ED: 0.017 mL), with no visual extension of T2/FLAIR signal abnormality into the adjacent white matter.

*This report is an automated research draft generated for research evaluation and must be reviewed and verified by a board-certified radiologist prior to any clinical utilization.*