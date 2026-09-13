"""Author/validate/adjudicate/freeze without ever importing a router executor."""
import argparse
import json
from pathlib import Path

from codeai.router_contract import adjudicate, freeze_manifest, validate_corpus, write_once

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "adjudicate", "freeze"))
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text())
    if args.command == "validate":
        print(f"Valid draft schema: {len(validate_corpus(corpus))} cases. NOT FROZEN.")
    else:
        if args.output is None:
            parser.error("--output required (write-once)")
        if args.command == "adjudicate":
            if args.adjudications is None:
                parser.error("--adjudications required")
            value = adjudicate(corpus, json.loads(args.adjudications.read_text()))
        else:
            if args.oracle is None or args.configuration is None:
                parser.error("--oracle and --configuration required")
            value = freeze_manifest(corpus, json.loads(args.oracle.read_text()),
                                    json.loads(args.configuration.read_text()))
        write_once(args.output, value)
