
from collections import defaultdict
import numpy as np


class RuleEngine:
    """Fast, master-data-driven product configuration engine."""

    def __init__(self, characteristic_values, configuration_rules):

        # Valid values for each characteristic
        self.all_options = {
            char: list(dict.fromkeys(group["VALUE"].tolist()))
            for char, group in characteristic_values.groupby(
                "CHARACTERISTIC", sort=False
            )
        }

        self.rules = []
        self.inc_rules = []

        # Compile Excel rules only once
        for rule_id, group in configuration_rules.groupby("RuleId"):

            conditions = defaultdict(set)
            actions = []

            for row in group.itertuples(index=False):
                if row.Operator == "Rule":
                    conditions[row.CHARACTERISTIC].add(row.VALUE)

                elif row.Operator == "Action":
                    actions.append(
                        (row.Type, row.CHARACTERISTIC, row.VALUE)
                    )

            inc_rows = group[group["Type"] == "INC"]

            if not inc_rows.empty:
                combination = {
                    row.CHARACTERISTIC: row.VALUE
                    for row in inc_rows.itertuples(index=False)
                }
                self.inc_rules.append((rule_id, combination))

            if actions:
                self.rules.append(
                    (rule_id, dict(conditions), actions)
                )

    @staticmethod
    def conditions_match(conditions, product):
        """OR within one characteristic, AND between characteristics."""

        return all(
            product.get(char) in allowed_values
            for char, allowed_values in conditions.items()
        )

    def get_options(self, characteristic, product):
        """Evaluate Auto Select, Enable and Disable rules."""

        if characteristic not in self.all_options:
            raise KeyError(f"Unknown characteristic: {characteristic}")

        options = set(self.all_options[characteristic])
        enabled = set()
        disabled = set()
        automatic = set()
        has_enable = False

        for rule_id, conditions, actions in self.rules:

            if not self.conditions_match(conditions, product):
                continue

            for action_type, target, value in actions:

                if target != characteristic:
                    continue

                if action_type == "Auto Select":
                    automatic.add(value)

                elif action_type == "Enable":
                    enabled.add(value)
                    has_enable = True

                elif action_type == "Disable":
                    disabled.add(value)

        if len(automatic) > 1:
            raise ValueError(
                f"Conflicting Auto Select rules for {characteristic}: "
                f"{automatic}"
            )

        if has_enable:
            options &= enabled

        options -= disabled

        if automatic:
            selected = next(iter(automatic))

            if selected not in self.all_options[characteristic]:
                raise ValueError(
                    f"Unknown Auto Select value: {characteristic}={selected}"
                )

            if selected in disabled:
                raise ValueError(
                    f"Auto Select / Disable conflict: {characteristic}"
                )

            # Auto Select has priority over Enable.
            # Its compatibility is checked separately.
            return [selected], selected

        return sorted(options), None

    def violates_inc(self, product):
        """Check INC combinations, including partially built products."""

        for rule_id, combination in self.inc_rules:

            if all(
                char in product and product[char] == value
                for char, value in combination.items()
            ):
                return True

        return False

    def choose_value(self, characteristic, product, rng, weights=None):
        """Choose a valid value with optional sampling weights."""

        options, automatic = self.get_options(
            characteristic, product
        )

        valid_options = [
            value for value in options
            if not self.violates_inc({
                **product,
                characteristic: value
            })
        ]

        if not valid_options:
            raise ValueError(
                f"No valid options for {characteristic}: {product}"
            )

        if automatic is not None:
            if automatic not in valid_options:
                raise ValueError(
                    f"Auto Select / INC conflict: {characteristic}"
                )
            return automatic

        if weights:
            probabilities = np.array(
                [weights.get(value, 0) for value in valid_options],
                dtype=float
            )

            if (probabilities < 0).any():
                raise ValueError("Sampling weights cannot be negative")

            if probabilities.sum() > 0:
                probabilities /= probabilities.sum()
                return rng.choice(valid_options, p=probabilities)

        return rng.choice(valid_options)

    def validate_product(self, product):
        """Validate a completed product against master rules."""

        errors = []

        if self.violates_inc(product):
            errors.append("INC incompatibility detected")

        for characteristic, selected in product.items():

            if characteristic not in self.all_options:
                continue

            if selected not in self.all_options[characteristic]:
                errors.append(
                    f"{characteristic}: unknown value {selected}"
                )
                continue

            options, _ = self.get_options(characteristic, product)

            if selected not in options:
                errors.append(
                    f"{characteristic}: {selected} violates selection rules"
                )

        return errors
