# Streamlit application — customer requirement review and product matching

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp_inference import NLPInference
from src.product_matching import ProductMatcher


st.set_page_config(
    page_title="AI-Assisted Oven Product Matching",
    page_icon="🔎",
    layout="wide"
)


# ==================================================
# LOAD NLP AND PRODUCT MATCHING ENGINES
# ==================================================

@st.cache_resource
def load_engines():
    return NLPInference(), ProductMatcher()


# ==================================================
# RESET CUSTOMER REQUEST
# ==================================================

def reset_request():

    for key in (
        "requirements",
        "analyzed_text",
        "matching_results",
        "confirmed_requirements",
        "requirement_editor"
    ):
        st.session_state.pop(key, None)

    st.session_state["customer_text"] = ""

    st.session_state["editor_version"] = (
        st.session_state.get("editor_version", 0) + 1
    )


# ==================================================
# GET VALID CHARACTERISTIC VALUES
# ==================================================

def valid_options(matcher, characteristic):

    if characteristic == "OVEN_FUNCTIONS":

        return sorted({
            str(v).strip()
            for values in matcher.function_mapping.values()
            for v in values
            if str(v).strip()
        })

    return sorted({
        str(v).strip()
        for v in matcher.products[characteristic].dropna().unique()
        if str(v).strip()
    })


# ==================================================
# LOAD CHARACTERISTIC-VALUE DICTIONARY
# ==================================================

@st.cache_data
def load_dictionary():

    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "characteristic_values.csv"
    )

    df = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        keep_default_na=False
    )

    df.columns = df.columns.str.strip().str.upper()

    expected = {
        "CHARACTERISTIC",
        "VALUE",
        "DEFINITION"
    }

    if not expected.issubset(df.columns):

        raise ValueError(
            f"Dictionary is missing columns: "
            f"{expected - set(df.columns)}"
        )

    df = df[list(sorted(expected))].copy()

    for col in expected:
        df[col] = df[col].astype(str).str.strip()

    if df.duplicated(
        ["CHARACTERISTIC", "VALUE"]
    ).any():

        raise ValueError(
            "Duplicate characteristic/value pairs in dictionary"
        )

    return {
        (r.CHARACTERISTIC, r.VALUE): r.DEFINITION
        for r in df.itertuples(index=False)
    }


# ==================================================
# CONVERT VALUE CODES INTO READABLE DEFINITIONS
# ==================================================

def display_value(characteristic, value):

    if pd.isna(value):
        return ""

    return VALUE_DICTIONARY.get(
        (
            str(characteristic).strip(),
            str(value).strip()
        ),
        str(value)
    )


# ==================================================
# PREPARE PRODUCT TABLE FOR DISPLAY
# ==================================================

def display_product_table(df):

    shown = df.copy()

    for column in shown.columns:

        if (
            column in matcher.products.columns
            and column != "PRODUCT_ID"
        ):

            shown[column] = shown[column].map(
                lambda v, c=column: display_value(c, v)
            )

    return shown


# ==================================================
# APPLICATION HEADER
# ==================================================

st.title("AI-Assisted Oven Product Matching")

st.caption(
    "NLP-based customer requirement extraction "
    "and rule-based product recommendation"
)

nlp, matcher = load_engines()

VALUE_DICTIONARY = load_dictionary()


# ==================================================
# SECTION 1 — CUSTOMER REQUEST
# ==================================================

st.subheader("1. Customer Request")

if "customer_text" not in st.session_state:

    st.session_state["customer_text"] = (
        "I'm looking for a steam oven "
        "that's exactly 45 cm wide."
    )

st.text_area(
    "Describe the oven you are looking for:",
    key="customer_text",
    height=110
)

left, right = st.columns([1, 1])

with left:

    analyze = st.button(
        "Analyze Request",
        type="primary"
    )

with right:

    st.button(
        "New Request",
        on_click=reset_request
    )


# ==================================================
# NLP PREDICTION
# ==================================================

if analyze:

    customer_text = st.session_state["customer_text"]

    if not customer_text.strip():

        st.warning(
            "Please enter a customer request."
        )

    else:

        try:

            predicted = nlp.predict_requirements(
                customer_text
            )

            predicted = predicted.copy()

            predicted["VALUE_DEFINITION"] = (
                predicted.apply(
                    lambda r: display_value(
                        r["CHARACTERISTIC"],
                        r["VALUE"]
                    ),
                    axis=1
                )
            )

            st.session_state["requirements"] = predicted

            st.session_state["analyzed_text"] = (
                customer_text
            )

            st.session_state.pop(
                "matching_results",
                None
            )

            st.session_state.pop(
                "confirmed_requirements",
                None
            )

            st.session_state["editor_version"] = (
                st.session_state.get(
                    "editor_version",
                    0
                ) + 1
            )

        except Exception as exc:

            st.error(
                f"Analysis failed: {exc}"
            )


# ==================================================
# SECTION 2 — REVIEW REQUIREMENTS
# ==================================================

if "requirements" in st.session_state:

    st.subheader(
        "2. Review Predicted Requirements"
    )

    st.info(
        "Review and correct the extracted requirements "
        "before matching products. "
        "Priority predictions may be inaccurate."
    )

    requirements = (
        st.session_state["requirements"].copy()
    )

    if not requirements.empty:

        requirements["VALUE_DEFINITION"] = (
            requirements.apply(
                lambda r: display_value(
                    r["CHARACTERISTIC"],
                    r["VALUE"]
                ),
                axis=1
            )
        )

    # --------------------------------------------------
    # Requirement editor
    # --------------------------------------------------

    if requirements.empty:

        st.warning(
            "No requirements were extracted. "
            "You can add requirements manually below."
        )

        edited = requirements.copy()

    else:

        # Hide VALUE from the interface.
        # Keep original codes internally for matching.

        visible_requirements = requirements.drop(
            columns=["VALUE"],
            errors="ignore"
        )

        edited_visible = st.data_editor(
            visible_requirements,
            hide_index=True,
            num_rows="fixed",
            disabled=[
                "CHARACTERISTIC",
                "VALUE_DEFINITION"
            ],
            column_config={

                "REQUIREMENT_TYPE":
                    st.column_config.SelectboxColumn(
                        "Priority",
                        options=[
                            "MUST",
                            "PREFERRED"
                        ],
                        required=True
                    ),

                "OPERATOR":
                    st.column_config.SelectboxColumn(
                        "Operator",
                        options=[
                            "EQUAL",
                            "CONTAINS"
                        ],
                        required=True
                    ),
            },
            key=(
                "requirement_editor_"
                f"{st.session_state.get('editor_version', 0)}"
            ),
            use_container_width=True
        )

        # Apply user edits to the original requirements.
        # VALUE codes remain unchanged.

        edited = requirements.copy()

        for editable_column in (
            "REQUIREMENT_TYPE",
            "OPERATOR"
        ):

            edited[editable_column] = (
                edited_visible[
                    editable_column
                ].to_numpy()
            )

    # --------------------------------------------------
    # Add missing requirement
    # --------------------------------------------------

    st.markdown(
        "**Add a missing requirement**"
    )

    excluded = {
        "PRODUCT_ID",
        "PREFERRED_SCORE"
    }

    characteristics = sorted(
        c
        for c in matcher.products.columns
        if c not in excluded
    )

    characteristic = st.selectbox(
        "Characteristic",
        characteristics
    )

    options = valid_options(
        matcher,
        characteristic
    )

    if options:

        value = st.selectbox(
            "Value / function",
            options,
            format_func=lambda v: display_value(
                characteristic,
                v
            )
        )

    else:

        value = None

    priority = st.selectbox(
        "Priority",
        [
            "MUST",
            "PREFERRED"
        ]
    )

    submitted = st.button(
        "Add Requirement"
    )

    # --------------------------------------------------
    # Save manually added requirement
    # --------------------------------------------------

    if submitted:

        if value is None:

            st.error(
                "No valid values found for this characteristic."
            )

        else:

            operator = (
                "CONTAINS"
                if characteristic == "OVEN_FUNCTIONS"
                else "EQUAL"
            )

            new_row = {
                "CHARACTERISTIC": characteristic,
                "VALUE": value,
                "OPERATOR": operator,
                "REQUIREMENT_TYPE": priority,
                "VALUE_DEFINITION": display_value(
                    characteristic,
                    value
                )
            }

            updated = pd.concat(
                [
                    edited,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )

            st.session_state["requirements"] = updated

            st.session_state.pop(
                "matching_results",
                None
            )

            st.session_state.pop(
                "confirmed_requirements",
                None
            )

            st.session_state["editor_version"] = (
                st.session_state.get(
                    "editor_version",
                    0
                ) + 1
            )

            st.rerun()

    # --------------------------------------------------
    # Confirm requirements and match products
    # --------------------------------------------------

    if st.button(
        "Confirm Requirements & Match Products"
    ):

        try:

            if edited.empty:

                st.warning(
                    "Add at least one requirement "
                    "before matching products."
                )

            else:

                matching_results = (
                    matcher.match_products(edited)
                )

                st.session_state[
                    "matching_results"
                ] = matching_results

                st.session_state[
                    "confirmed_requirements"
                ] = edited.copy()

        except Exception as exc:

            st.error(
                f"Product matching failed: {exc}"
            )


# ==================================================
# SECTION 3 — RECOMMENDED PRODUCTS
# ==================================================

if "matching_results" in st.session_state:

    st.subheader(
        "3. Recommended Products"
    )

    results = (
        st.session_state["matching_results"]
    )

    if results.empty:

        st.warning(
            "No existing product satisfies "
            "all MUST requirements."
        )

        st.info(
            "A new product configuration may be required. "
            "Rule-based configuration generation is planned "
            "for a future development phase."
        )

    else:

        st.success(
            f"{len(results)} eligible products found."
        )

        preferred_count = (
            st.session_state[
                "confirmed_requirements"
            ]["REQUIREMENT_TYPE"]
            .eq("PREFERRED")
            .sum()
        )

        st.caption(
            "Preferred score: number of satisfied "
            f"preferences (maximum {preferred_count})."
        )

        display_columns = [
            "PRODUCT_ID",
            "PREFERRED_SCORE",
            "OVEN_TYPE",
            "WIDTH_CM",
            "OVEN_FUNCTIONS",
            "OVEN_VOLUME_L",
            "ENERGY_CLASS"
        ]

        display_columns = [
            col
            for col in display_columns
            if col in results.columns
        ]

        st.dataframe(
            display_product_table(
                results[display_columns].head(50)
            ),
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Download Matching Products (CSV)",
            data=results.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="matching_products.csv",
            mime="text/csv"
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Academic prototype | "
    "NLP 1: TF-IDF + Linear SVM | "
    "NLP 2: TF-IDF + Logistic Regression | "
    "Product matching: deterministic "
    "MUST/PREFERRED rules"
)