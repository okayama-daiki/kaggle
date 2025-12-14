# Santa 2025 - Christmas Tree Packing Challenge

## Evaluation

Submissions are evaluated on sum of the normalized area of the square bounding box for each puzzle. For each `n`-tree configuration, the side `s` of square box bounding the trees is squared and divided by the total number `n` of trees in the configuration. The final score is the sum of all configurations. Refer to the metric notebook for exact implementation details.

$$
\text{score} = \sum_{n=1}^{N} \frac{s_n^2}{n}
$$

## Submission file

For each `id` in the submission (representing a single tree in a $n$-tree configuration), you must report the tree position given by `x`, `y`, and the rotation given by `deg`. To avoid loss of precision when saving and reading the files, the values must be converted to a string and prepended with an `s` before submission. Submissions with any overlapping trees will throw an error. To avoid extreme leaderboard scores, location values must be constrained to $-100 \le x, y \le 100$.

The file should contain a header and have the following format:

```txt
id,x,y,deg
001_0,s0.0,s0.0,s20.411299
002_0,s0.0,s0.0,s20.411299
002_1,s-0.541068,s0.259317,s51.66348
etc.
```

```bash
uvx kaggle competitions submit -c santa-2025 -f submission.csv -m "Your submission message here"
```

## Data

The objective of this challenge is to arrange Christmas tree toys into the smallest packing arrangement possible—as defined by the size of a square bounding box around the trees—for toy counts of 1-200.

```bash
uvx kaggle competitions download -c santa-2025
unzip santa-2025.zip && rm santa-2025.zip
```
