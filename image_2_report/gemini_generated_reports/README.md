# Gemini MRI Report Generation

Research prototype for generating structured MRI observations
using the Gemini API.

## Setup

1. Activate your Python environment.
2. Install requirements:
   pip install -r requirements.txt
3. Add your Gemini API key to .env.
4. Test the API:
   python scripts/test_gemini_connection.py
5. Generate a report:
   python scripts/generate_one_patient.py --patient BraTSPED_00001

Input:
../data/cache_128/<patient_id>.npz

Outputs:
outputs/
visualizations/

This prototype is not for clinical use without institutional
approval, validation, and expert review.
