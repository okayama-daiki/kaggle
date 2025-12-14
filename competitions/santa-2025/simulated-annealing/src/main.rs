use anyhow::{Context, Result};
use clap::Parser;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;
use std::fs::{self, File};
use std::path::PathBuf;
use std::time::{Duration, Instant};

// Tree vertices (non-convex), matching Kaggle metric notebook
const TREE_POLYGON: &[(f64, f64)] = &[
    (0.0, 0.8),
    (0.125, 0.5),
    (0.0625, 0.5),
    (0.2, 0.25),
    (0.1, 0.25),
    (0.35, 0.0),
    (0.075, 0.0),
    (0.075, -0.2),
    (-0.075, -0.2),
    (-0.075, 0.0),
    (-0.35, 0.0),
    (-0.1, 0.25),
    (-0.2, 0.25),
    (-0.0625, 0.5),
    (-0.125, 0.5),
];

#[derive(Parser, Debug)]
#[command(name = "Santa SA")]
#[command(about = "Simulated annealing optimizer for Christmas tree packing", long_about = None)]
struct Args {
    /// Submission CSV file to read initial layout from
    #[arg(short, long)]
    file: PathBuf,

    /// Number of trees to optimize (if 0, optimize all 1-200)
    #[arg(short = 'n', long, default_value_t = 0)]
    num_trees: usize,

    /// Time limit in seconds per tree count (default 30)
    #[arg(long, default_value_t = 30)]
    time: u64,

    /// Max iterations (if no time provided)
    #[arg(long, default_value_t = 10000)]
    max_iter: usize,

    /// Output CSV path for improved solution
    #[arg(short, long, default_value = "submission_optimized.csv")]
    output: PathBuf,

    /// Progress print interval (seconds)
    #[arg(long, default_value_t = 5)]
    print_interval: u64,

    /// Penalty weight for overlap (larger discourages overlap)
    #[arg(long, default_value_t = 10.0)]
    penalty_weight: f64,

    /// Initial temperature
    #[arg(long, default_value_t = 1.0)]
    t0: f64,

    /// Cooling factor per iteration (0<alpha<1)
    #[arg(long, default_value_t = 0.9995)]
    alpha: f64,

    /// Step sigma for move proposals
    #[arg(long, default_value_t = 0.05)]
    step_sigma: f64,

    /// How many nearby trees to move together (1 = single-tree move)
    #[arg(long, default_value_t = 1)]
    group_size: usize,
}

#[derive(Debug, Deserialize, Clone)]
struct TreeRecord {
    id: String,
    x: String,
    y: String,
    deg: String,
}

#[derive(Debug, Clone)]
struct Tree {
    x: f64,
    y: f64,
    angle: f64,
}

impl Tree {
    fn from_record(record: TreeRecord) -> Result<Self> {
        let x_str = record.x.trim_start_matches('s');
        let y_str = record.y.trim_start_matches('s');
        let deg_str = record.deg.trim_start_matches('s');
        Ok(Tree {
            x: x_str.parse().context("parse x")?,
            y: y_str.parse().context("parse y")?,
            angle: deg_str.parse().context("parse angle")?,
        })
    }
}

fn read_initial(file: &PathBuf, n: usize) -> Result<Vec<Tree>> {
    let f = File::open(file).context("open file")?;
    let mut rdr = csv::ReaderBuilder::new().flexible(true).from_reader(f);
    let mut trees = Vec::new();
    let prefix = format!("{:03}_", n);
    for rec in rdr.deserialize::<TreeRecord>() {
        match rec {
            Ok(r) => {
                if r.id.starts_with(&prefix) {
                    trees.push(Tree::from_record(r)?);
                }
            }
            Err(_) => continue,
        }
    }
    if trees.len() != n {
        anyhow::bail!("Expected {} trees, found {}", n, trees.len());
    }
    Ok(trees)
}

fn rotate_and_translate(tree: &Tree) -> Vec<(f64, f64)> {
    let rad = tree.angle.to_radians();
    let (sin, cos) = rad.sin_cos();
    TREE_POLYGON
        .iter()
        .map(|(px, py)| {
            let rx = px * cos - py * sin + tree.x;
            let ry = px * sin + py * cos + tree.y;
            (rx, ry)
        })
        .collect()
}

fn polygon_bounds(poly: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let mut minx = f64::INFINITY;
    let mut miny = f64::INFINITY;
    let mut maxx = f64::NEG_INFINITY;
    let mut maxy = f64::NEG_INFINITY;
    for (x, y) in poly.iter().copied() {
        minx = minx.min(x);
        maxx = maxx.max(x);
        miny = miny.min(y);
        maxy = maxy.max(y);
    }
    (minx, miny, maxx, maxy)
}

fn bounding_side_polygons(trees: &[Vec<(f64, f64)>]) -> f64 {
    let mut minx = f64::INFINITY;
    let mut miny = f64::INFINITY;
    let mut maxx = f64::NEG_INFINITY;
    let mut maxy = f64::NEG_INFINITY;
    for poly in trees {
        let (px0, py0, px1, py1) = polygon_bounds(poly);
        minx = minx.min(px0);
        miny = miny.min(py0);
        maxx = maxx.max(px1);
        maxy = maxy.max(py1);
    }
    (maxx - minx).max(maxy - miny)
}

fn segments_intersect(a1: (f64, f64), a2: (f64, f64), b1: (f64, f64), b2: (f64, f64)) -> bool {
    fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    }
    let d1 = cross(a1, a2, b1);
    let d2 = cross(a1, a2, b2);
    let d3 = cross(b1, b2, a1);
    let d4 = cross(b1, b2, a2);

    if (d1 == 0.0 && d2 == 0.0 && d3 == 0.0 && d4 == 0.0) {
        // Collinear: check overlap in projection
        let (ax_min, ax_max) = if a1.0 <= a2.0 {
            (a1.0, a2.0)
        } else {
            (a2.0, a1.0)
        };
        let (ay_min, ay_max) = if a1.1 <= a2.1 {
            (a1.1, a2.1)
        } else {
            (a2.1, a1.1)
        };
        let (bx_min, bx_max) = if b1.0 <= b2.0 {
            (b1.0, b2.0)
        } else {
            (b2.0, b1.0)
        };
        let (by_min, by_max) = if b1.1 <= b2.1 {
            (b1.1, b2.1)
        } else {
            (b2.1, b1.1)
        };
        return ax_min <= bx_max && bx_min <= ax_max && ay_min <= by_max && by_min <= ay_max;
    }

    (d1 >= 0.0 && d2 <= 0.0 || d1 <= 0.0 && d2 >= 0.0)
        && (d3 >= 0.0 && d4 <= 0.0 || d3 <= 0.0 && d4 >= 0.0)
}

fn point_in_poly(p: (f64, f64), poly: &[(f64, f64)]) -> bool {
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        let intersect =
            ((yi > p.1) != (yj > p.1)) && (p.0 < (xj - xi) * (p.1 - yi) / (yj - yi + 1e-12) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn polygons_intersect(a: &[(f64, f64)], b: &[(f64, f64)]) -> bool {
    for i in 0..a.len() {
        let a1 = a[i];
        let a2 = a[(i + 1) % a.len()];
        for j in 0..b.len() {
            let b1 = b[j];
            let b2 = b[(j + 1) % b.len()];
            if segments_intersect(a1, a2, b1, b2) {
                return true;
            }
        }
    }
    point_in_poly(a[0], b) || point_in_poly(b[0], a)
}

// Objective: bounding square side length for tree polygon, plus overlap penalties.
fn objective(trees: &[Tree], penalty_w: f64) -> f64 {
    if trees.is_empty() {
        return 0.0;
    }

    let mut polys = Vec::with_capacity(trees.len());
    for t in trees {
        polys.push(rotate_and_translate(t));
    }

    let side = bounding_side_polygons(&polys);

    let mut penalty = 0.0;
    for i in 0..polys.len() {
        for j in (i + 1)..polys.len() {
            if polygons_intersect(&polys[i], &polys[j]) {
                penalty += penalty_w;
            }
        }
    }

    side + penalty
}

fn write_csv(output: &PathBuf, n: usize, trees: &[Tree]) -> Result<()> {
    let mut wtr = csv::Writer::from_path(output)?;
    wtr.write_record(["id", "x", "y", "deg"])?;
    // Rows must match Kaggle format with 's' prefix
    for (i, t) in trees.iter().enumerate() {
        let id = format!("{:03}_{}", n, i);
        wtr.write_record([
            id,
            format!("s{:.6}", t.x),
            format!("s{:.6}", t.y),
            format!("s{:.6}", t.angle),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

fn write_aggregated_csv(output: &PathBuf, all_results: &[(usize, Vec<Tree>)]) -> Result<()> {
    let mut wtr = csv::Writer::from_path(output)?;
    wtr.write_record(["id", "x", "y", "deg"])?;

    for (n, trees) in all_results {
        for (i, t) in trees.iter().enumerate() {
            let id = format!("{:03}_{}", n, i);
            wtr.write_record([
                id,
                format!("s{:.6}", t.x),
                format!("s{:.6}", t.y),
                format!("s{:.6}", t.angle),
            ])?;
        }
    }
    wtr.flush()?;
    Ok(())
}

fn compute_leaderboard_score(
    all_results: &[(usize, Vec<Tree>)],
) -> Result<(f64, Vec<(usize, f64, usize)>)> {
    let mut total = 0.0;
    let mut per_n = Vec::new();

    for (n, trees) in all_results {
        if trees.len() != *n {
            anyhow::bail!("Expected {} trees for n={}, found {}", n, n, trees.len());
        }
        let polys: Vec<Vec<(f64, f64)>> = trees.iter().map(rotate_and_translate).collect();
        let side = bounding_side_polygons(&polys);
        let score = side * side / *n as f64;
        per_n.push((*n, score, trees.len()));
        total += score;
    }

    per_n.sort_by_key(|(n, _, _)| *n);
    Ok((total, per_n))
}

fn anneal(mut trees: Vec<Tree>, args: &Args, _n: usize) -> Vec<Tree> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, args.step_sigma).unwrap();

    let mut best = trees.clone();
    let mut best_cost = objective(&best, args.penalty_weight);

    let mut current = trees;
    let mut current_cost = best_cost;

    let mut t = args.t0;
    let start = Instant::now();
    let deadline = start + Duration::from_secs(args.time);
    let mut last_print = start;

    let mut iter = 0usize;
    loop {
        if Instant::now() >= deadline {
            break;
        }

        // Propose move: pick a seed tree and move it along with its nearest neighbors
        let seed = rng.gen_range(0..current.len());
        let k = args.group_size.max(1).min(current.len());

        // Gather indices of k-1 closest trees to the seed (plus the seed itself)
        let mut dists: Vec<(usize, f64)> = current
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let dx = t.x - current[seed].x;
                let dy = t.y - current[seed].y;
                (i, dx * dx + dy * dy)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut indices: Vec<usize> = dists.into_iter().take(k).map(|(i, _)| i).collect();
        if !indices.contains(&seed) {
            indices[0] = seed; // ensure seed is included
        }

        let dx = normal.sample(&mut rng);
        let dy = normal.sample(&mut rng);
        let dtheta = normal.sample(&mut rng) * 10.0; // degrees

        // Rotate around group centroid, then translate
        let mut proposal = current.clone();
        let (cx, cy) = {
            let mut sumx = 0.0;
            let mut sumy = 0.0;
            for &i in &indices {
                sumx += proposal[i].x;
                sumy += proposal[i].y;
            }
            (sumx / indices.len() as f64, sumy / indices.len() as f64)
        };

        let rad = dtheta.to_radians();
        let (s, c) = rad.sin_cos();
        for &i in &indices {
            let px = proposal[i].x - cx;
            let py = proposal[i].y - cy;
            let rx = px * c - py * s;
            let ry = px * s + py * c;
            proposal[i].x = rx + cx + dx;
            proposal[i].y = ry + cy + dy;
            proposal[i].angle = (proposal[i].angle + dtheta) % 360.0;
        }

        let prop_cost = objective(&proposal, args.penalty_weight);
        let delta = prop_cost - current_cost;
        let accept = delta < 0.0 || rng.gen::<f64>() < (-delta / t).exp();
        if accept {
            current = proposal;
            current_cost = prop_cost;
            if current_cost < best_cost {
                best = current.clone();
                best_cost = current_cost;
            }
        }

        // Cooling
        t *= args.alpha;
        iter += 1;

        // Progress
        let now = Instant::now();
        if now.duration_since(last_print) >= Duration::from_secs(args.print_interval) {
            println!("  iter={} t={:.4} best={:.6}", iter, t, best_cost);
            last_print = now;
        }
    }

    println!("Finished: iter={} best_cost={:.6}", iter, best_cost);
    best
}

fn main() -> Result<()> {
    let args = Args::parse();

    let range: Vec<usize> = if args.num_trees == 0 {
        (1..=200).collect()
    } else {
        vec![args.num_trees]
    };

    println!(
        "Processing {} tree count(s) from {:?}",
        range.len(),
        args.file
    );

    let mut all_results = Vec::new();

    for n in range {
        println!("\n=== Optimizing n={} ===", n);

        match read_initial(&args.file, n) {
            Ok(init) => {
                println!(
                    "Loaded {} trees, starting {} second annealing...",
                    init.len(),
                    args.time
                );
                let best = anneal(init, &args, n);
                all_results.push((n, best));
                println!("✓ Completed n={}", n);
            }
            Err(e) => {
                eprintln!("✗ Failed n={}: {}", n, e);
                continue;
            }
        }
    }

    // Aggregate results into single submission CSV
    println!(
        "\nWriting {} results to {:?}...",
        all_results.len(),
        args.output
    );
    write_aggregated_csv(&args.output, &all_results)?;

    let (score, _) = compute_leaderboard_score(&all_results)?;
    let score_filename = format!("{:.6}.csv", score);
    let scored_output = PathBuf::from("submissions").join(score_filename);
    if let Some(parent) = scored_output.parent() {
        fs::create_dir_all(parent)?;
    }
    write_aggregated_csv(&scored_output, &all_results)?;

    println!("Leaderboard score: {:.6}", score);
    println!("Saved scored submission to {:?}", scored_output);
    println!("Done.");
    Ok(())
}
