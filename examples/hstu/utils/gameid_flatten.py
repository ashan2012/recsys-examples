#!/usr/bin/env python3
# Copyright (c) 2025
#
# Given input lines in the format:
#   user_id \t item_id,timestamp ; item_id,timestamp ; ...
# produce a CSV with columns item_id,user_id,timestamp

import argparse
import csv
import logging
import sys


def parse_line(line: str):
    """
    Parse a single line of the custom format.

    Returns:
        user_id (str), list[tuple[item_id, timestamp]]
    """
    user_part, interactions_part = line.rstrip("\n").split("\t")
    interactions = []
    if interactions_part:
        for token in interactions_part.split(";"):
            token = token.strip()
            if not token:
                continue
            item_id, ts = token.split(",")
            interactions.append((item_id.strip(), ts.strip()))
    return user_part.strip(), interactions


def convert_file(input_path: str, output_path: str, log_interval: int):
    logging.info("Starting conversion from %s to %s", input_path, output_path)
    num_users = 0
    num_interactions = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8", newline=""
    ) as fout:
        writer = csv.writer(fout)
        writer.writerow(["item_id", "user_id", "timestamp"])
        for line in fin:
            line = line.strip()
            if not line:
                continue
            user_id, interactions = parse_line(line)
            num_users += 1
            for item_id, ts in interactions:
                writer.writerow([item_id, user_id, ts])
                num_interactions += 1
            if log_interval > 0 and num_users % log_interval == 0:
                logging.info(
                    "Processed %d users / %d interactions ...",
                    num_users,
                    num_interactions,
                )
    logging.info(
        "Finished conversion. Total users=%d, interactions=%d",
        num_users,
        num_interactions,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert custom GameID text format to CSV."
    )
    parser.add_argument("--input", required=True, help="Input txt file path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100000,
        help="Log progress every N users (default: 100000).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    convert_file(args.input, args.output, args.log_interval)


if __name__ == "__main__":
    main()


