# STEP 192 — Product Matching Module
# Based on Notebook 05, Steps 165–176

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


class ProductMatcher:

    def __init__(self):

        # Load Product Master
        self.products = pd.read_csv(
            DATA_DIR / "product_master.csv",
            dtype=str
        )

        # Normalize characteristic values
        for column in self.products.columns:
            self.products[column] = (
                self.products[column].astype("string").str.strip()
            )

        assert self.products["PRODUCT_ID"].is_unique

        # Load oven function combinations
        function_set = pd.read_csv(
            DATA_DIR / "oven_function_set.csv",
            header=None,
            dtype=str,
            encoding="utf-8-sig"
        )

        self.function_mapping = {}

        for _, row in function_set.iterrows():

            code = row.iloc[1]

            if pd.isna(code):
                continue

            code = str(code).strip()

            if not code or code.lower() == "nan":
                continue

            functions = [
                str(value).strip()
                for value in row.iloc[4:]
                if pd.notna(value) and str(value).strip()
            ]

            self.function_mapping[code] = functions

        # Validate function coverage
        product_codes = set(
            self.products["OVEN_FUNCTIONS"].dropna()
        )

        missing = product_codes - set(self.function_mapping)

        assert not missing, (
            f"Unmapped oven function codes: {missing}"
        )

        # Expand function combinations for each product
        self.product_function_sets = {
            product_id: set(self.function_mapping.get(code, []))
            for product_id, code in zip(
                self.products["PRODUCT_ID"],
                self.products["OVEN_FUNCTIONS"]
            )
        }

    def requirement_mask(self, requirement):
        """
        Evaluate one requirement against all products.
        """

        characteristic = str(
            requirement["CHARACTERISTIC"]
        ).strip()

        operator = str(
            requirement["OPERATOR"]
        ).strip().upper()

        value = str(
            requirement["VALUE"]
        ).strip()

        if characteristic not in self.products.columns:
            raise ValueError(
                f"Unknown characteristic: {characteristic}"
            )

        if operator == "EQUAL":

            return (
                self.products[characteristic] == value
            ).fillna(False)

        elif operator == "CONTAINS":

            if characteristic != "OVEN_FUNCTIONS":
                raise ValueError(
                    f"Unsupported CONTAINS characteristic: "
                    f"{characteristic}"
                )

            return self.products["PRODUCT_ID"].map(
                lambda pid: (
                    value in self.product_function_sets[pid]
                )
            )

        else:
            raise ValueError(
                f"Unsupported operator: {operator}"
            )

    def match_products(self, requirements):
        """
        Filter by MUST and rank by PREFERRED.

        requirements: DataFrame containing
            CHARACTERISTIC
            VALUE
            OPERATOR
            REQUIREMENT_TYPE
        """

        required_columns = {
            "CHARACTERISTIC",
            "VALUE",
            "OPERATOR",
            "REQUIREMENT_TYPE"
        }

        missing_columns = required_columns - set(
            requirements.columns
        )

        if missing_columns:
            raise ValueError(
                f"Missing requirement columns: {missing_columns}"
            )

        priorities = (
            requirements["REQUIREMENT_TYPE"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        if priorities.isna().any() or not priorities.isin(
            ["MUST", "PREFERRED"]
        ).all():
            raise ValueError(
                "REQUIREMENT_TYPE must be MUST or PREFERRED"
            )

        must_reqs = requirements.loc[
            priorities == "MUST"
        ]

        preferred_reqs = requirements.loc[
            priorities == "PREFERRED"
        ]

        # Stage 1: MUST filtering
        eligible_mask = pd.Series(
            True,
            index=self.products.index
        )

        for _, requirement in must_reqs.iterrows():
            eligible_mask &= self.requirement_mask(requirement)

        eligible_products = self.products.loc[
            eligible_mask
        ].copy()

        # Stage 2: PREFERRED scoring
        eligible_products["PREFERRED_SCORE"] = 0

        for _, requirement in preferred_reqs.iterrows():

            preference_mask = self.requirement_mask(
                requirement
            )

            eligible_products["PREFERRED_SCORE"] += (
                preference_mask.loc[
                    eligible_products.index
                ].astype(int)
            )

        eligible_products = eligible_products.sort_values(
            ["PREFERRED_SCORE", "PRODUCT_ID"],
            ascending=[False, True]
        ).reset_index(drop=True)

        return eligible_products