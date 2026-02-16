import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


CLASS_MAP = {"CNV": "0", "DME": "1", "DRUSEN": "2", "NORMAL": "3"}


def run_cmd(cmd, log_path=None):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text())
    return default


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def discover_checkpoints(diff_runs_root: Path, diff_name: str):
    ckpts = []
    for run_dir in sorted(diff_runs_root.glob(f"{diff_name}_*")):
        models = run_dir / "models"
        if not models.exists():
            continue
        for p in sorted(models.glob("diff_*.pt")):
            m = re.match(r"diff_(\d+)\.pt$", p.name)
            if not m:
                continue
            idx = int(m.group(1))
            epoch = idx + 1
            ckpts.append((epoch, run_dir, p))
    ckpts.sort(key=lambda x: (x[0], str(x[2])))
    return ckpts


def link_or_copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for class_name, class_idx in CLASS_MAP.items():
        sdir = src / class_name
        ddir = dst / class_idx
        ddir.mkdir(parents=True, exist_ok=True)
        for img in sdir.glob("*.png"):
            target = ddir / img.name
            try:
                target.hardlink_to(img)
            except Exception:
                shutil.copy2(img, target)


def parse_all_fid(report_text: str):
    all_line = None
    for line in report_text.splitlines():
        if line.strip().startswith("ALL") and "|" in line:
            all_line = line
    if all_line is None:
        return None
    parts = [p.strip() for p in all_line.split("|")]
    if len(parts) < 7:
        return None
    try:
        return {
            "fid_all": float(parts[3]),
            "is_all": float(parts[4].split("(")[0].strip()),
            "ssim_all": float(parts[5]),
        }
    except Exception:
        return None


def is_diffusion_training_active(diff_config: Path):
    cmd = [
        "python3",
        "-c",
        (
            "import subprocess;"
            "out=subprocess.check_output(['ps','-ef'], text=True);"
            f"print('YES' if '{str(diff_config)}' in out and 'train_diffusion.py' in out else 'NO')"
        ),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return out.endswith("YES")


def append_csv(row, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp",
        "epoch",
        "checkpoint",
        "fid_all",
        "is_all",
        "ssim_all",
        "samples_per_class",
        "steps",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Monitor diffusion checkpoints and evaluate balanced FID")
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--diff_runs_root", required=True)
    ap.add_argument("--diff_name", default="diff_fraction_consistent")
    ap.add_argument("--diff_config", required=True)
    ap.add_argument("--real_root", required=True)
    ap.add_argument("--eval_every_epochs", type=int, default=100)
    ap.add_argument("--samples_per_class", type=int, default=250)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poll_seconds", type=int, default=180)
    ap.add_argument("--stop_when_idle", action="store_true")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    diff_runs_root = Path(args.diff_runs_root)
    diff_config = Path(args.diff_config)
    real_root = Path(args.real_root)

    state_path = run_root / "logs" / "diff_fid_monitor_state.json"
    csv_path = run_root / "logs" / "diff_fid_monitor.csv"
    monitor_tmp = run_root / "generated" / "_fid_monitor_tmp"
    report_tmp_dir = run_root / "metrics_reports" / "_fid_monitor_tmp"
    monitor_log_dir = run_root / "logs" / "diff_fid_monitor_runs"
    monitor_log_dir.mkdir(parents=True, exist_ok=True)

    state = load_json(
        state_path,
        {
            "processed": [],
            "best": None,
        },
    )
    processed = set(state.get("processed", []))
    idle_rounds = 0

    while True:
        ckpts = discover_checkpoints(diff_runs_root, args.diff_name)
        pending = []
        for epoch, run_dir, ckpt in ckpts:
            if epoch % int(args.eval_every_epochs) != 0:
                continue
            key = str(ckpt)
            if key in processed:
                continue
            pending.append((epoch, run_dir, ckpt))

        if pending:
            idle_rounds = 0
        else:
            idle_rounds += 1

        for epoch, run_dir, ckpt in pending:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"eval_e{epoch}_{stamp}"
            gen_root = monitor_tmp / dataset_name
            eval_root = monitor_tmp / f"{dataset_name}_eval_0to3"
            gen_log = monitor_log_dir / f"gen_epoch{epoch}_{stamp}.log"
            fid_log = monitor_log_dir / f"fid_epoch{epoch}_{stamp}.log"

            cfg = json.loads(diff_config.read_text())
            vae_ckpt = cfg.get("main", {}).get("trained_autoencoder_path")
            if not vae_ckpt:
                raise RuntimeError("main.trained_autoencoder_path is missing in diffusion config")

            gen_cmd = [
                "conda",
                "run",
                "-n",
                "maisi",
                "python",
                str(Path(__file__).resolve().parent / "generate_diffusion_dataset.py"),
                "--out_root",
                str(monitor_tmp),
                "--dataset_name",
                dataset_name,
                "--vae_ckpt",
                str(vae_ckpt),
                "--diff_ckpt",
                str(ckpt),
                "--diff_config",
                str(diff_config),
                "--steps",
                str(args.steps),
                "--batch_size",
                str(args.batch_size),
                "--fixed_per_class",
                str(args.samples_per_class),
                "--seed",
                str(args.seed),
            ]
            run_cmd(gen_cmd, gen_log)

            link_or_copy_tree(gen_root, eval_root)

            report_tmp_dir.mkdir(parents=True, exist_ok=True)
            existing_reports = set(report_tmp_dir.glob("*.txt"))
            fid_cmd = [
                "conda",
                "run",
                "-n",
                "maisi",
                "python",
                str(Path(__file__).resolve().parent.parent / "compute_oct_metrics.py"),
                "--real_root",
                str(real_root),
                "--fake_root",
                str(eval_root),
                "--out_dir",
                str(report_tmp_dir),
            ]
            out = run_cmd(fid_cmd, fid_log)
            parsed = parse_all_fid(out)
            new_reports = set(report_tmp_dir.glob("*.txt")) - existing_reports
            for rp in new_reports:
                try:
                    rp.unlink()
                except Exception:
                    pass

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "epoch": epoch,
                "checkpoint": str(ckpt),
                "fid_all": parsed["fid_all"] if parsed else "",
                "is_all": parsed["is_all"] if parsed else "",
                "ssim_all": parsed["ssim_all"] if parsed else "",
                "samples_per_class": int(args.samples_per_class),
                "steps": int(args.steps),
            }
            append_csv(row, csv_path)

            if parsed is not None:
                best = state.get("best")
                if best is None or float(parsed["fid_all"]) < float(best.get("fid_all", 1e9)):
                    state["best"] = {
                        "epoch": epoch,
                        "checkpoint": str(ckpt),
                        "fid_all": float(parsed["fid_all"]),
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    }

            processed.add(str(ckpt))
            state["processed"] = sorted(processed)
            save_json(state_path, state)

            # Save space: remove generated eval images immediately.
            for p in [gen_root, eval_root]:
                if p.exists():
                    shutil.rmtree(p)

        if args.stop_when_idle:
            active = is_diffusion_training_active(diff_config)
            if (not active) and (not pending) and idle_rounds >= 2:
                break

        time.sleep(int(args.poll_seconds))


if __name__ == "__main__":
    main()
