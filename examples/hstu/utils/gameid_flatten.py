#!/usr/bin/env python3
# Copyright (c) 2025
#
# Given input lines in the format:
#   user_id \t item_id,timestamp ; item_id,timestamp ; ...
# produce a CSV with columns item_id,user_id,timestamp

import argparse
import csv


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


def convert_file(input_path: str, output_path: str):
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
            for item_id, ts in interactions:
                writer.writerow([item_id, user_id, ts])


def main():
    parser = argparse.ArgumentParser(
        description="Convert custom GameID text format to CSV."
    )
    parser.add_argument("--input", required=True, help="Input txt file path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()
    convert_file(args.input, args.output)


if __name__ == "__main__":
    main()


