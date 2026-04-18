import argparse
from pathlib import Path

from src.experiments.grid import expand_sweep_spec, load_sweep_spec, write_expanded_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Generate concrete experiment configs from a sweep spec.")
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path to the sweep spec JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the output directory.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the generated experiment names without writing files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    spec = load_sweep_spec(args.spec)
    configs = expand_sweep_spec(spec)

    if args.print_only:
        for config_payload in configs:
            print(config_payload["experiment_name"])
        return

    output_dir = args.output_dir or spec.output_dir
    written_paths = write_expanded_configs(configs, output_dir=output_dir)
    print(f"Generated {len(written_paths)} configs in {output_dir}")
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
