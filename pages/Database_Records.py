import streamlit as st
import pandas as pd
from database import get_reports


st.set_page_config(
    page_title="Database Records",
    layout="wide"
)

st.title("OHE Thermal Inspection Database")
st.caption("Historical and current thermal inspection analysis records")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
try:
    reports = get_reports(limit=1000)
except Exception as e:
    st.error(f"Unable to load database records: {e}")
    st.stop()


if not reports:
    st.info("No inspection records found in the database.")
    st.stop()


df = pd.DataFrame(reports)


# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Filters")


# Section filter
if "section" in df.columns:
    sections = sorted(
        [str(x) for x in df["section"].dropna().unique()]
    )

    selected_sections = st.sidebar.multiselect(
        "Section",
        sections
    )

    if selected_sections:
        df = df[df["section"].astype(str).isin(selected_sections)]


# Status filter
if "status" in df.columns:
    statuses = sorted(
        [str(x) for x in df["status"].dropna().unique()]
    )

    selected_statuses = st.sidebar.multiselect(
        "Status",
        statuses
    )

    if selected_statuses:
        df = df[df["status"].astype(str).isin(selected_statuses)]


# Image / mast search
search_text = st.sidebar.text_input(
    "Search image / OHE mast"
)

if search_text:
    search_text = search_text.lower()

    image_match = (
        df["image_name"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(search_text, na=False)
        if "image_name" in df.columns
        else False
    )

    mast_match = (
        df["ohe_mast"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(search_text, na=False)
        if "ohe_mast" in df.columns
        else False
    )

    df = df[image_match | mast_match]


# ---------------------------------------------------------
# DISPLAY RECORD COUNT
# ---------------------------------------------------------
st.write(f"**Records found:** {len(df)}")


# ---------------------------------------------------------
# DISPLAY TABLE
# ---------------------------------------------------------
display_columns = [
    "timestamp",
    "image_name",
    "capture_date",
    "capture_time",
    "section",
    "ohe_mast",
    "scale_max",
    "scale_min",
    "wire_max",
    "wire_min",
    "delta_t",
    "status",
    "attend_in",
]

display_columns = [
    col for col in display_columns
    if col in df.columns
]

st.dataframe(
    df[display_columns],
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# CSV DOWNLOAD
# ---------------------------------------------------------
csv_data = df[display_columns].to_csv(index=False)

st.download_button(
    label="Download Records as CSV",
    data=csv_data,
    file_name="KRCL_OHE_Thermal_Records.csv",
    mime="text/csv"
)
