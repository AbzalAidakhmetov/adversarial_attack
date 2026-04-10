#!/usr/bin/env python3
"""Build norm-matched and random baseline steering vectors."""

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create baseline steering-vector variants from a saved attack output."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the steering_vector.pt file containing clean and poisoned vectors.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the random baseline vectors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input_path, map_location="cpu", weights_only=False)

    clean = payload["steering_vector_clean"]
    poisoned = payload["steering_vector_poisoned"]

    clean_norm = clean.norm()
    poisoned_norm = poisoned.norm()
    ratio = poisoned_norm / clean_norm
    print(
        f"  Clean norm: {clean_norm:.4f}, Poisoned norm: {poisoned_norm:.4f}, "
        f"Ratio: {ratio:.2f}x"
    )

    normed_payload = dict(payload)
    normed_payload["steering_vector_poisoned"] = poisoned * (clean_norm / poisoned_norm)
    torch.save(normed_payload, args.input_path.with_name("steering_vector_normed.pt"))

    torch.manual_seed(args.seed)
    random_vector = torch.randn_like(poisoned)
    random_vector = random_vector * (poisoned_norm / random_vector.norm())

    random_payload = dict(payload)
    random_payload["steering_vector_poisoned"] = random_vector
    torch.save(random_payload, args.input_path.with_name("steering_vector_random.pt"))

    random_normed_payload = dict(payload)
    random_normed_payload["steering_vector_poisoned"] = (
        random_vector * (clean_norm / random_vector.norm())
    )
    torch.save(
        random_normed_payload,
        args.input_path.with_name("steering_vector_random_normed.pt"),
    )


if __name__ == "__main__":
    main()
