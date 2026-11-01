# STEP 193C — NLP Inference Module
# Based on Notebook 04 (Steps 151, 154, 180)
# and Notebook 05 (Steps 183, 185, 188D, 188E)

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

REQUIREMENT_COLUMNS = [
    "CHARACTERISTIC",
    "VALUE",
    "OPERATOR",
    "REQUIREMENT_TYPE",
]


class NLPInference:

    def __init__(self):

        # Load trained NLP models
        nlp1 = joblib.load(
            MODEL_DIR / "nlp1_requirement_extraction.joblib"
        )

        nlp2 = joblib.load(
            MODEL_DIR / "nlp2_priority_classification.joblib"
        )

        self.vectorizer = nlp1["vectorizer"]
        self.model = nlp1["model"]
        self.labels = list(nlp1["labels"])
        self.threshold = float(nlp1["threshold"])

        self.priority_pipeline = nlp2["pipeline"]
        self.class_mapping = nlp2["class_mapping"]

        # Load characteristic-value descriptions
        descriptions = pd.read_csv(
            DATA_DIR / "nlp_ground_truth.csv",
            dtype=str
        )

        descriptions = (
            descriptions[
                ["CHARACTERISTIC", "VALUE", "VALUE_DEFINITION"]
            ]
            .dropna()
            .drop_duplicates()
        )

        # Check for conflicting descriptions
        conflicts = (
            descriptions
            .groupby(["CHARACTERISTIC", "VALUE"])[
                "VALUE_DEFINITION"
            ]
            .nunique()
        )

        assert (conflicts <= 1).all(), (
            "Conflicting value descriptions detected."
        )

        self.description_map = {
            (row.CHARACTERISTIC, row.VALUE): row.VALUE_DEFINITION
            for row in descriptions.itertuples(index=False)
        }

        assert len(self.labels) == 99
        assert self.threshold == -0.3

    def predict_requirements(self, customer_text):
        """
        NLP 1: Extract characteristic-value requirements.
        NLP 2: Predict MUST/PREFERRED for each requirement.
        """

        customer_text = str(customer_text).strip()

        if not customer_text:
            raise ValueError("Customer request cannot be empty.")

        # ------------------------------------------
        # NLP 1 — Requirement extraction
        # ------------------------------------------

        X = self.vectorizer.transform([customer_text])

        scores = np.asarray(
            self.model.decision_function(X)
        ).reshape(-1)

        assert len(scores) == len(self.labels)

        selected_labels = [
            label
            for label, score in zip(self.labels, scores)
            if score >= self.threshold
        ]

        records = []

        for label in selected_labels:

            characteristic, value = label.rsplit("__", 1)

            operator = (
                "CONTAINS"
                if characteristic == "OVEN_FUNCTIONS"
                else "EQUAL"
            )

            description = self.description_map.get(
                (characteristic, value)
            )

            if description is None:
                raise ValueError(
                    f"Missing description: {characteristic}={value}"
                )

            records.append({
                "CHARACTERISTIC": characteristic,
                "VALUE": value,
                "OPERATOR": operator,
                "VALUE_DEFINITION": description,
            })

        if not records:
            return pd.DataFrame(
                columns=REQUIREMENT_COLUMNS
            )

        requirements = pd.DataFrame(records)

        # ------------------------------------------
        # NLP 2 — Priority classification
        # ------------------------------------------

        requirements["REQUIREMENT_DESCRIPTION"] = (
            requirements["CHARACTERISTIC"]
            + " = "
            + requirements["VALUE_DEFINITION"]
        )

        # Preserve the exact training input format
        requirements["NLP2_TEXT"] = (
            "CUSTOMER REQUEST: "
            + customer_text
            + " REQUIREMENT: "
            + requirements["REQUIREMENT_DESCRIPTION"]
        )

        predictions = self.priority_pipeline.predict(
            requirements["NLP2_TEXT"].tolist()
        )

        requirements["REQUIREMENT_TYPE"] = [
            self.class_mapping[int(prediction)]
            for prediction in predictions
        ]

        return requirements[
            [
                "CHARACTERISTIC",
                "VALUE",
                "OPERATOR",
                "REQUIREMENT_TYPE",
                "VALUE_DEFINITION",
            ]
        ].copy()