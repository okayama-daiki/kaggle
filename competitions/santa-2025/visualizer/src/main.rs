use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use csv::ReaderBuilder;
use plotters::prelude::*;
use serde::Deserialize;

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
#[command(
    name = "Santa ScoreViz",
    about = "Score and visualize Santa 2025 submissions"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compute leaderboard score from a submission CSV
    Score {
        /// Submission CSV path
        #[arg(short, long)]
        file: PathBuf,
        /// Print per-n breakdown
        #[arg(long, default_value_t = false)]
        per_n: bool,
        /// Only score up to this n (optional)
        #[arg(long)]
        max_n: Option<usize>,
    },
    /// Render a single n to PNG
    Render {
        /// Submission CSV path
        #[arg(short, long)]
        file: PathBuf,
        /// Tree count to render
        #[arg(short, long)]
        n: usize,
        /// Output PNG path
        #[arg(short, long, default_value = "render.png")]
        output: PathBuf,
        /// Image width/height in pixels (square)
        #[arg(long, default_value_t = 800)]
        size: u32,
        /// Margin as fraction of bounding side (e.g., 0.1 = 10%)
        #[arg(long, default_value_t = 0.1)]
        margin: f64,
    },
}

#[derive(Deserialize)]
struct Record {
    id: String,
    x: String,
    y: String,
    deg: String,
}

#[derive(Clone, Copy, Debug)]
struct Tree {
    x: f64,
    y: f64,
    deg: f64,
}

fn strip_prefix(s: &str) -> &str {
    s.trim_start_matches('s')
}

fn parse_tree(rec: Record) -> Result<(usize, Tree)> {
    let (n_str, _) = rec
        .id
        .split_once('_')
        .ok_or_else(|| anyhow!("Invalid id: {}", rec.id))?;
    let n: usize = n_str.parse().context("parse n")?;
    let x: f64 = strip_prefix(&rec.x).parse().context("parse x")?;
    let y: f64 = strip_prefix(&rec.y).parse().context("parse y")?;
    let deg: f64 = strip_prefix(&rec.deg).parse().context("parse deg")?;
    Ok((n, Tree { x, y, deg }))
}

fn rotate_and_translate(tree: &Tree) -> Vec<(f64, f64)> {
    let rad = tree.deg.to_radians();
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
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in poly.iter().copied() {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    (min_x, max_x, min_y, max_y)
}

fn load_submission(path: &PathBuf, max_n: Option<usize>) -> Result<HashMap<usize, Vec<Tree>>> {
    let file = File::open(path).with_context(|| format!("open submission: {:?}", path))?;
    let mut rdr = ReaderBuilder::new().flexible(true).from_reader(file);
    let mut map: HashMap<usize, Vec<Tree>> = HashMap::new();
    for rec in rdr.deserialize::<Record>() {
        let rec = rec?;
        let (n, tree) = parse_tree(rec)?;
        if let Some(limit) = max_n {
            if n > limit {
                continue;
            }
        }
        map.entry(n).or_default().push(tree);
    }
    Ok(map)
}

fn bounding_side(trees: &[Tree]) -> (f64, f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for poly in trees.iter().map(rotate_and_translate) {
        let (px0, px1, py0, py1) = polygon_bounds(&poly);
        min_x = min_x.min(px0);
        max_x = max_x.max(px1);
        min_y = min_y.min(py0);
        max_y = max_y.max(py1);
    }

    let side = (max_x - min_x).max(max_y - min_y);
    (side, min_x, max_x, min_y, max_y)
}

fn compute_score(
    groups: &HashMap<usize, Vec<Tree>>,
    max_n: Option<usize>,
) -> Result<(f64, Vec<(usize, f64, usize)>)> {
    let mut total = 0.0;
    let mut per_n = Vec::new();
    for (&n, trees) in groups.iter() {
        if let Some(limit) = max_n {
            if n > limit {
                continue;
            }
        }
        if trees.len() != n {
            return Err(anyhow!(
                "Expected {} trees for n={}, found {}",
                n,
                n,
                trees.len()
            ));
        }
        let (side, _, _, _, _) = bounding_side(trees);
        let score = side * side / n as f64;
        per_n.push((n, score, trees.len()));
        total += score;
    }
    per_n.sort_by_key(|(n, _, _)| *n);
    Ok((total, per_n))
}

fn render_group(
    trees: &[Tree],
    n: usize,
    side: f64,
    center_x: f64,
    center_y: f64,
    size: u32,
    margin: f64,
    output: &PathBuf,
) -> Result<()> {
    let margin_abs = side * margin;
    let extent = side / 2.0 + margin_abs;
    let x0 = center_x - extent;
    let x1 = center_x + extent;
    let y0 = center_y - extent;
    let y1 = center_y + extent;

    let root = BitMapBackend::new(output, (size, size)).into_drawing_area();
    root.fill(&WHITE)?;
    let w = (x1 - x0).max(1e-9);
    let h = (y1 - y0).max(1e-9);
    let to_px = |wx: f64, wy: f64| {
        let px = ((wx - x0) / w * (size as f64)) as i32;
        let py = ((y1 - wy) / h * (size as f64)) as i32; // invert y so up is up
        (px, py)
    };

    // Bounding square outline in pixel space
    let bb_min_world = (center_x - side / 2.0, center_y - side / 2.0);
    let bb_max_world = (center_x + side / 2.0, center_y + side / 2.0);
    let bb_min_px = to_px(bb_min_world.0, bb_min_world.1);
    let bb_max_px = to_px(bb_max_world.0, bb_max_world.1);
    root.draw(&Rectangle::new([bb_min_px, bb_max_px], RED.stroke_width(2)))?;

    // Trees as small circles in pixel space
    for (i, t) in trees.iter().enumerate() {
        let color = Palette99::pick(i);
        let poly = rotate_and_translate(t);
        let screen_pts: Vec<_> = poly.iter().map(|(x, y)| to_px(*x, *y)).collect();
        root.draw(&Polygon::new(screen_pts.clone(), color.mix(0.5).filled()))?;
        root.draw(&PathElement::new(screen_pts, color.stroke_width(1)))?;
    }

    root.present()?;
    println!("Rendered n={} to {:?}", n, output);
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Score { file, per_n, max_n } => {
            let groups = load_submission(&file, max_n)?;
            let (total, per_group) = compute_score(&groups, max_n)?;
            if per_n {
                for (n, score, count) in per_group {
                    println!("n={}: score={:.6} ({} trees)", n, score, count);
                }
            }
            println!("Total score: {:.6}", total);
        }
        Command::Render {
            file,
            n,
            output,
            size,
            margin,
        } => {
            let groups = load_submission(&file, Some(n))?;
            let trees = groups
                .get(&n)
                .ok_or_else(|| anyhow!("n={} not found in submission", n))?;
            if trees.len() != n {
                return Err(anyhow!(
                    "Expected {} trees for n={}, found {}",
                    n,
                    n,
                    trees.len()
                ));
            }
            let (side, min_x, max_x, min_y, max_y) = bounding_side(trees);
            let center_x = (min_x + max_x) / 2.0;
            let center_y = (min_y + max_y) / 2.0;
            render_group(trees, n, side, center_x, center_y, size, margin, &output)?;
            println!("Rendered n={} to {:?}", n, output);
        }
    }

    Ok(())
}
