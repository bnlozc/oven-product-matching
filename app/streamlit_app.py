# STEP 194A — Streamlit Application
# AI-Assisted Oven Product Matching

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
    layout="wide",
)


@st.cache_resource
def load_engines():
    return NLPInference(), ProductMatcher()


st.title("AI-Assisted Oven Product Matching")

st.caption(
    "NLP-based customer requirement extraction "
    "and rule-based product recommendation"
)

nlp, matcher = load_engines()


# --------------------------------------------
# 1. Customer request
# --------------------------------------------

st.subheader("1. Customer Request")

customer_text = st.text_area(
    "Describe the oven you are looking for:",
    value="I'm looking for a steam oven that's exactly 45 cm wide.",
    height=110,
)


if st.button("Analyze Request", type="primary"):

    if not customer_text.strip():
        st.warning("Please enter a customer request.")

    else:
        try:
            predicted = nlp.predict_requirements(customer_text)

            st.session_state["requirements"] = predicted
            st.session_state["analyzed_text"] = customer_text
            st.session_state.pop("matching_results", None)

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")


# --------------------------------------------
# 2. Requirement review
# --------------------------------------------

if "requirements" in st.session_state:

    st.subheader("2. Review Predicted Requirements")

    st.info(
        "Review and correct the extracted requirements "
        "before matching products. "
        "Priority predictions may be inaccurate."
    )

    requirements = st.session_state["requirements"]

    if requirements.empty:

        st.warning(
            "No requirements were extracted. "
            "Please revise the customer request."
        )

    else:

        edited = st.data_editor(
            requirements,
            hide_index=True,
            num_rows="fixed",
            disabled=[
                "CHARACTERISTIC",
                "VALUE",
                "VALUE_DEFINITION",
            ],
            column_config={
                "REQUIREMENT_TYPE": st.column_config.SelectboxColumn(
                    "Priority",
                    options=["MUST", "PREFERRED"],
                    required=True,
                ),
                "OPERATOR": st.column_config.SelectboxColumn(
                    "Operator",
                    options=["EQUAL", "CONTAINS"],
                    required=True,
                ),
            },
            key="requirement_editor",
            use_container_width=True,
        )

        if st.button("Confirm Requirements & Match Products"):

            try:

                matching_results = matcher.match_products(edited)

                st.session_state["matching_results"] = (
                    matching_results
                )

                st.session_state["confirmed_requirements"] = (
                    edited.copy()
                )

            except Exception as exc:
                st.error(f"Product matching failed: {exc}")


# --------------------------------------------
# 3. Product matching results
# --------------------------------------------

if "matching_results" in st.session_state:

    st.subheader("3. Recommended Products")

    results = st.session_state["matching_results"]

    if results.empty:

        st.warning(
            "No existing product satisfies all MUST requirements."
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
            st.session_state["confirmed_requirements"][
                "REQUIREMENT_TYPE"
            ]
            .eq("PREFERRED")
            .sum()
        )

        st.caption(
            f"Preferred score: number of satisfied preferences "
            f"(maximum {preferred_count})."
        )

        display_columns = [
            "PRODUCT_ID",
            "PREFERRED_SCORE",
            "OVEN_TYPE",
            "WIDTH_CM",
            "OVEN_FUNCTIONS",
            "OVEN_VOLUME_L",
            "ENERGY_CLASS",
        ]

        display_columns = [
            col for col in display_columns
            if col in results.columns
        ]

        st.dataframe(
            results[display_columns].head(50),
            hide_index=True,
            use_container_width=True,
        )

        csv_data = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Matching Products (CSV)",
            data=csv_data,
            file_name="matching_products.csv",
            mime="text/csv",
        )


st.divider()

st.caption(
    "Academic prototype | "
    "NLP 1: TF-IDF + Linear SVM | "
    "NLP 2: TF-IDF + Logistic Regression | "
    "Product matching: deterministic MUST/PREFERRED rules"
)