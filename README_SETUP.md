# KRCL AI-Assisted OHE Thermal Inspection

## Architecture

Thermal image
-> OCR thermal scale
-> Gemini Vision OHE segmentation
-> local colour-to-temperature mapping
-> robust max/min temperature
-> Delta T
-> configured status
-> Supabase report

Gemini does NOT calculate the temperature. Python does.

## Files

- app.py
- thermal_logic.py
- gemini_vision.py
- database.py
- pages/Database_Records.py
- requirements.txt
- packages.txt
- supabase_schema.sql
- .streamlit/secrets.toml

## Setup

1. Create a Gemini API key.
2. Create a Supabase project/table.
3. Run supabase_schema.sql.
4. Put secrets in .streamlit/secrets.toml.
5. Install dependencies:
   pip install -r requirements.txt
6. Make sure Tesseract is installed.
7. Run:
   streamlit run app.py

## Important

Do not use streamlit-drawable-canvas.
There is no manual rectangle/ROI selection in this version.

The system intentionally stops when:
- the thermal scale cannot be read,
- the scale is inconsistent,
- Gemini returns no OHE mask,
- the AI mask is implausibly small/large,
- too few valid thermal pixels remain.

It does not silently substitute fake/default temperatures.
