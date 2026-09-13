"""Read-only reanalysis of a routing ledger; no adapters, credentials or runner."""
import argparse
import json

from codeai.router_analysis import verify_ledger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger")
    parser.add_argument("artifacts")
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(json.dumps(verify_ledger(args.ledger, args.artifacts, args.run_id), indent=2))
