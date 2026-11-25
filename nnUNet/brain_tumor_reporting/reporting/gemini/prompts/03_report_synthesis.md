# Brain MRI Research Report Synthesis

Prepare a structured research-draft brain MRI report using:
1. The structured quantitative case summary.
2. The image interpretation produced in the first stage.
3. The source and evidence limitations described below.

## Evidence rules

- Original MRI observations are the primary imaging evidence.
- Segmentation masks are model predictions and must not be treated as ground truth.
- Quantitative tumor volumes must be taken from the supplied case summary.
- Do not recalculate or invent numerical values.
- Do not add TC and WT to their component volumes; they are composite totals.
- Distinguish percent_tumor_in_anatomy from percent_anatomy_occupied.
- Anatomical overlap is spatial overlap after image alignment, not proof of histological invasion.
- SynthSeg anatomical boundaries may be distorted by mass effect, edema, or segmentation error.
- SynthSeg regional QC scores are quality-control measures, not diagnostic confidence scores.
- ET, NET, CC and ED are computational segmentation labels.
- Do not infer enhancement solely from the ET label.
- Do not infer tumor histology, molecular subtype, grade, or prognosis.
- Do not invent demographics, clinical history, symptoms, acquisition parameters, pathology, or treatment history.
- If information is unavailable, state that it was not provided.
- If visual interpretation and quantitative measurements appear inconsistent, report the discrepancy for review rather than resolving it by guessing.

## Required report structure

# Brain MRI Report — Research Draft

## 1. Examination and available data
Describe the available MRI sequences and computational analyses. Do not invent acquisition details.

## 2. MRI findings
Summarize visible findings from the original MRI images, organized by sequence where appropriate.
Distinguish observations from uncertain interpretations.

## 3. Segmentation findings
Report ET, NET, CC, ED, TC and WT volumes exactly as supplied.
Identify absent components by their zero measured volume.
Explain that these are automated segmentation-derived measurements.

## 4. Anatomical spatial overlap
Summarize the most relevant anatomical overlap findings.
Use the correct percentage definitions:
- percent_tumor_in_anatomy: fraction of the tumor component overlapping that anatomical region.
- percent_anatomy_occupied: fraction of the anatomical region overlapping the tumor component.

Explicitly state that spatial overlap does not establish histological infiltration or invasion.

## 5. Image and segmentation quality
Summarize available visual QC and SynthSeg QC information.
Do not treat QC scores as diagnostic findings.

## 6. Impression
Provide a concise, evidence-based summary.
Separate directly visible MRI observations from segmentation-derived measurements.
Do not provide a histological diagnosis or grade.

## 7. Limitations
Mention missing clinical and demographic information, segmentation uncertainty, anatomical registration limitations, and the need for radiologist review as applicable.

## 8. Review checklist
List important observations or uncertainties that require human verification.

## Style
- Use concise, professional radiology-style language.
- Avoid repetitive descriptions.
- Use units consistently (mL for volumes).
- Preserve the distinction between original MRI observations and model-derived measurements.
- Never present the draft as a final clinical report.
- If evidence is insufficient, say so explicitly.
