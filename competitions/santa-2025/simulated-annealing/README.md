# Santa 2025 - Simulated Annealing (Rust)

A simple simulated annealing optimizer that adjusts tree positions (as discs of radius 0.5) to reduce the required bounding square side while penalizing overlaps.

## Build

```bash
cd competitions/santa-2025/sa
cargo build --release
```

## Run

```bash
# Optimize n=50 trees from an existing submission CSV
./target/release/santa_sa \
  --file ../../competitions/santa-2025/submissions/71.97.txt \
  --num-trees 50 \
  --time 60 \
  --output improved_50.csv
```

- `--time`: time limit in seconds (alternatively use `--max-iter`)
- `--print-interval`: logs progress every N seconds (default 30)
- `--penalty-weight`: overlap penalty weight (default 10.0)
- `--step-sigma`: proposal step sigma (default 0.05)

## Notes

- The geometry is approximated by discs of radius 0.5 around tree centers.
- Overlaps are penalized quadratically.
- Rotation `deg` is kept and randomly perturbed but not used in the objective.
- Output CSV uses Kaggle submission format with `'s'` prefixes.
