# Step 99 - Resumable LLM customer request generation

import argparse
import csv
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data/processed/request_ground_truth_3000.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data/interim/llm_generated_requests.csv"
)

FIELDNAMES = [
    "REQUEST_ID",
    "LLM_TEXT",
    "INPUT_TOKENS",
    "OUTPUT_TOKENS",
    "REASONING_TOKENS",
    "MODEL",
]

SYSTEM_PROMPT = """
You write realistic English customer requests for household ovens.

Convert the structured requirements into one natural customer message.

Rules:
- Preserve every supplied requirement.
- Do not add any new product requirements.
- MUST means mandatory.
- PREFERRED means optional.
- CONTAINS means the oven includes the specified function.
- CONTAINS does not mean the oven has only that function.
- EQUAL means the characteristic must match the specified value.
- Do not expose internal characteristic names or technical codes.
- Use natural, varied English.
- Return only the customer message.
"""


def load_completed_ids():
    if not OUTPUT_FILE.exists():
        return set()

    existing = pd.read_csv(OUTPUT_FILE, dtype=str)

    if existing.empty:
        return set()

    return set(
        existing.loc[
            existing["LLM_TEXT"].notna()
            & existing["LLM_TEXT"].str.strip().ne(""),
            "REQUEST_ID"
        ]
    )


def save_result(record):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    file_exists = OUTPUT_FILE.exists()

    with OUTPUT_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)
        file.flush()
        os.fsync(file.fileno())


def generate_request(client, request_id, requirements):

    user_prompt = (
        "Write one natural English customer request "
        "using these requirements:\n"
        + json.dumps(requirements, ensure_ascii=False)
    )

    response = client.responses.create(
        model="gpt-5-mini",
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        reasoning={"effort": "minimal"},
        max_output_tokens=1000,
    )

    text = response.output_text.strip()

    if response.status != "completed" or not text:
        raise RuntimeError(
            f"Incomplete response for {request_id}: "
            f"{response.status}"
        )

    reasoning_tokens = (
        response.usage.output_tokens_details.reasoning_tokens
        if response.usage.output_tokens_details
        else 0
    )

    return {
        "REQUEST_ID": request_id,
        "LLM_TEXT": text,
        "INPUT_TOKENS": response.usage.input_tokens,
        "OUTPUT_TOKENS": response.usage.output_tokens,
        "REASONING_TOKENS": reasoning_tokens,
        "MODEL": "gpt-5-mini",
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=3
    )

    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found")

    client = OpenAI(
        timeout=30.0,
        max_retries=1
    )

    ground_truth = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        keep_default_na=False
    )

    completed_ids = load_completed_ids()

    print("Total requests:",
          ground_truth["REQUEST_ID"].nunique())

    print("Already completed:", len(completed_ids))

    generated = 0

    for request_id, group in ground_truth.groupby(
        "REQUEST_ID",
        sort=True
    ):

        if request_id in completed_ids:
            continue

        requirements = group[
            [
                "CHARACTERISTIC",
                "VALUE_DEFINITION",
                "OPERATOR",
                "REQUIREMENT_TYPE"
            ]
        ].to_dict(orient="records")

        try:
            result = generate_request(
                client,
                request_id,
                requirements
            )

            save_result(result)

            generated += 1

            print(
                f"Saved {request_id} | "
                f"Input: {result['INPUT_TOKENS']} | "
                f"Output: {result['OUTPUT_TOKENS']}",
                flush=True
            )

        except Exception as error:
            print(
                f"ERROR {request_id}: {error}",
                flush=True
            )

            # Stop instead of spending credits repeatedly
            break

        if generated >= args.limit:
            break

        time.sleep(0.3)

    print("\nNew requests generated:", generated)
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()