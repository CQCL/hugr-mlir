#!/usr/bin/env python3

import argparse
from argparse import Namespace
from pathlib import Path
import subprocess

def main(args: Namespace) -> ():
    infile = Path(args.input_file)
    outdir = Path(args.output_dir) if args.output_dir else infile.parent

    if infile.suffix == ".msgpack":
        outfile = infile.with_suffix(".json")
    else:
        raise Exception(f"suffix not msgpack: {infile}")

    p = subprocess.run(["rq", "-m", "--output-json", "--format", "indented"], input=infile.read_bytes(), check=True, capture_output=True)
    Path(outfile).write_bytes(p.stdout)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('--output-dir', required = False)
    return parser

if __name__ == "__main__":
    main(make_parser().parse_args())
