# -*- coding: utf-8 -*-
"""
Server-side Supabase access.

SUPABASE_SECRET_KEY is preferred because the Streamlit server is the
trusted backend. Never expose this key in browser code or GitHub.
"""

import streamlit as st
from supabase import create_client


def _secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return None


SUPABASE_URL = _secret("SUPABASE_URL")

# Prefer the current server-side secret key.
SUPABASE_KEY = (
    _secret("SUPABASE_SECRET_KEY")
    or _secret("SUPABASE_KEY")
)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Supabase configuration is missing. Add SUPABASE_URL and "
        "SUPABASE_SECRET_KEY to Streamlit secrets."
    )


supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY,
)


def save_report(data):
    allowed_columns = {
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
    }

    clean_data = {
        key: value
        for key, value in data.items()
        if key in allowed_columns
    }

    return (
        supabase
        .table("thermal_results")
        .insert(clean_data)
        .execute()
    )


def get_reports(limit=1000):
    response = (
        supabase
        .table("thermal_results")
        .select("*")
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )

    return response.data or []
