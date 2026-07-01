"""
Driver Traits Dashboard

Lets the user pick a driver and see their 6 FM-style traits as a radar
chart plus individual score bars, similar to a player attributes screen.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.traits import load_traits
from src.data_pipeline import load_raw_data, prepare_data
from config import TRAIT_NAMES, TRAIT_COLORS, get_accuracy_caption
from components.trait_visualizer import render_trait_radar, render_trait_bars, render_multi_driver_radar

st.set_page_config(page_title="Driver Traits - F1 Race Predictor", page_icon="👤", layout="wide")

st.title("👤 Driver Traits")
st.caption("Traits reflect each driver's most recent seasons of form, not their entire career.")
st.caption(get_accuracy_caption())


@st.cache_data
def load_data():
    raw = load_raw_data()
    prepared = prepare_data(raw)
    traits = load_traits()
    return prepared, traits, raw['drivers']


prepared, traits, drivers_df = load_data()

recent_year = prepared['year'].max()
active_ids = set(prepared[prepared['year'] >= recent_year - 1]['driverId'].unique())
active_ids &= set(traits.keys())

id_to_name = {
    int(row.driverId): f"{row.forename} {row.surname}"
    for row in drivers_df.itertuples()
}

show_all = st.sidebar.checkbox("Include retired / historical drivers")
candidate_ids = set(traits.keys()) if show_all else active_ids

name_to_id = {id_to_name[d]: d for d in candidate_ids if d in id_to_name}
selected_name = st.sidebar.selectbox("Driver", sorted(name_to_id.keys()))
driver_id = name_to_id[selected_name]
driver_traits = traits[driver_id]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(selected_name)
    render_trait_radar(selected_name, driver_traits, TRAIT_NAMES)

with col2:
    st.subheader("Trait Breakdown")
    render_trait_bars(driver_traits, TRAIT_NAMES, TRAIT_COLORS)

st.markdown("---")
st.subheader("Compare Drivers")

compare_names = st.multiselect(
    "Select drivers to compare",
    sorted(name_to_id.keys()),
    default=[selected_name],
)

if compare_names:
    drivers_traits = {name: traits[name_to_id[name]] for name in compare_names}
    render_multi_driver_radar(drivers_traits, TRAIT_NAMES)

    rows = []
    for name in compare_names:
        d_id = name_to_id[name]
        row = {"Driver": name}
        row.update({TRAIT_NAMES.get(t, t): round(s, 2) for t, s in traits[d_id].items()})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("Driver"), use_container_width=True)
