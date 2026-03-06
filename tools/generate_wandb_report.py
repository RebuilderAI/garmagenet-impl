#!/usr/bin/env python3
"""Generate wandb Reports for GarmentGen training runs.

Usage:
    # VAE report
    python tools/generate_wandb_report.py --type vae --run-name vae_ncs_mask_lazy_npy

    # LDM report
    python tools/generate_wandb_report.py --type ldm --run-name garmagenet_text_cond

    # With custom title
    python tools/generate_wandb_report.py --type vae --run-id s5ei9dux --title "VAE Epoch 100 Results"

Requires: pip install wandb-workspaces
"""
import argparse
import sys

try:
    import wandb_workspaces.reports.v2 as wr
except ImportError:
    print("Error: wandb-workspaces not installed. Run: pip install wandb-workspaces")
    sys.exit(1)

ENTITY_DEFAULT = "jeremiah91-rebuilderai"
PROJECT_DEFAULT = "GarmentGen"
VAL_TIMESTEPS = [10, 50, 100, 200, 500, 1000]

# PanelGrid is 24 units wide. Layout(x, y, w, h) positions panels.
# Full width = w=24, half = w=12, third = w=8


def build_runset(entity: str, project: str, run_name: str = None, run_id: str = None) -> wr.Runset:
    """Build a Runset that filters to a specific wandb run."""
    display = run_name or run_id or "All Runs"
    kwargs = dict(entity=entity, project=project, name=display)
    if run_name:
        kwargs["filters"] = f"Name = '{run_name}'"
    elif run_id:
        kwargs["filters"] = f"ID = '{run_id}'"
    return wr.Runset(**kwargs)


def build_vae_blocks(runset: wr.Runset) -> list:
    """Build report blocks for VAE training — PDF-friendly layout.

    Layout strategy for PDF export:
    - Images: full width (w=24) stacked vertically — avoids tiny side-by-side panels
    - Charts: 2-column (w=12) — readable at PDF scale
    - Panel heights increased (h=10) for legibility
    """
    blocks = []

    # --- Section 1: Validation Visualizations (Geometry) ---
    blocks.append(wr.H2(text="Validation — Geometry (NCS)"))
    blocks.append(wr.P(text="Ground Truth input vs VAE Reconstruction output (logged every 5 epochs)."))
    blocks.append(wr.PanelGrid(
        runsets=[runset],
        panels=[
            wr.MediaBrowser(media_keys=["Val-Geo-Input"], num_columns=1,
                            layout=wr.Layout(x=0, y=0, w=24, h=8)),
            wr.MediaBrowser(media_keys=["Val-Geo"], num_columns=1,
                            layout=wr.Layout(x=0, y=8, w=24, h=8)),
        ],
    ))

    # --- Section 2: Validation Visualizations (Mask) ---
    blocks.append(wr.H2(text="Validation — Mask"))
    blocks.append(wr.PanelGrid(
        runsets=[runset],
        panels=[
            wr.MediaBrowser(media_keys=["Val-Mask-Input"], num_columns=1,
                            layout=wr.Layout(x=0, y=0, w=24, h=8)),
            wr.MediaBrowser(media_keys=["Val-Mask"], num_columns=1,
                            layout=wr.Layout(x=0, y=8, w=24, h=8)),
        ],
    ))

    # --- Section 3: Reconstruction Loss (2-column) ---
    blocks.append(wr.H2(text="Reconstruction Loss"))
    blocks.append(wr.PanelGrid(
        runsets=[runset],
        panels=[
            wr.LinePlot(title="Train MSE Loss", y=["loss-mse"], smoothing_show_original=True,
                        layout=wr.Layout(x=0, y=0, w=12, h=10)),
            wr.LinePlot(title="Val MSE Loss", y=["Val-mse"], smoothing_show_original=True,
                        layout=wr.Layout(x=12, y=0, w=12, h=10)),
            wr.LinePlot(title="KL Divergence", y=["loss-kl"], smoothing_show_original=True,
                        layout=wr.Layout(x=0, y=10, w=24, h=10)),
        ],
    ))

    # --- Section 4: Latent Space Statistics (2-column) ---
    blocks.append(wr.H2(text="Latent Space Statistics"))
    blocks.append(wr.PanelGrid(
        runsets=[runset],
        panels=[
            wr.LinePlot(title="z Mean", y=["z-mean"], smoothing_show_original=True,
                        layout=wr.Layout(x=0, y=0, w=12, h=10)),
            wr.LinePlot(title="z Std", y=["z-std"], smoothing_show_original=True,
                        layout=wr.Layout(x=12, y=0, w=12, h=10)),
            wr.LinePlot(title="Mode Std", y=["mode-std"], smoothing_show_original=True,
                        layout=wr.Layout(x=0, y=10, w=24, h=10)),
        ],
    ))

    return blocks


def build_ldm_blocks(runset: wr.Runset) -> list:
    """Build report blocks for LDM (GarmageNet) training."""
    blocks = []

    # --- Section 1: Training Losses ---
    blocks.append(wr.H2(text="Training Losses"))
    blocks.append(wr.PanelGrid(
        runsets=[runset],
        panels=[
            wr.LinePlot(title="Total Loss", y=["total_loss"], smoothing_show_original=True,
                        layout=wr.Layout(x=0, y=0, w=8, h=8)),
            wr.LinePlot(title="Latent Loss", y=["loss_latent"], smoothing_show_original=True,
                        layout=wr.Layout(x=8, y=0, w=8, h=8)),
            wr.LinePlot(title="BBox Loss", y=["loss_bbox"], smoothing_show_original=True,
                        layout=wr.Layout(x=16, y=0, w=8, h=8)),
        ],
    ))

    # --- Section 2: Val Latent MSE by Timestep ---
    latent_keys = [f"val-latent-{t:04d}" for t in VAL_TIMESTEPS]
    blocks.append(wr.H2(text="Validation Latent MSE by Timestep"))
    blocks.append(wr.PanelGrid(
        runsets=[runset],
        panels=[
            wr.LinePlot(title="Val Latent MSE (all timesteps)", y=latent_keys, smoothing_show_original=True,
                        layout=wr.Layout(x=0, y=0, w=24, h=10)),
        ],
    ))

    # --- Section 3: Val BBox MSE by Timestep ---
    bbox_keys = [f"val-bbox-{t:04d}" for t in VAL_TIMESTEPS]
    blocks.append(wr.H2(text="Validation BBox MSE by Timestep"))
    blocks.append(wr.PanelGrid(
        runsets=[runset],
        panels=[
            wr.LinePlot(title="Val BBox MSE (all timesteps)", y=bbox_keys, smoothing_show_original=True,
                        layout=wr.Layout(x=0, y=0, w=24, h=10)),
        ],
    ))

    # --- Section 4: Val BBox L1 Metrics ---
    bbox_l1_keys = [f"val-bbox-L1-{t:04d}" for t in VAL_TIMESTEPS]
    bbox_pw_l1_keys = [f"val-bbox_pw-L1-{t:04d}" for t in VAL_TIMESTEPS]
    blocks.append(wr.H2(text="Validation BBox L1 Metrics"))
    blocks.append(wr.PanelGrid(
        runsets=[runset],
        panels=[
            wr.LinePlot(title="Val BBox L1 (all timesteps)", y=bbox_l1_keys, smoothing_show_original=True,
                        layout=wr.Layout(x=0, y=0, w=12, h=10)),
            wr.LinePlot(title="Val BBox Per-Panel L1 (all timesteps)", y=bbox_pw_l1_keys, smoothing_show_original=True,
                        layout=wr.Layout(x=12, y=0, w=12, h=10)),
        ],
    ))

    return blocks


def main():
    parser = argparse.ArgumentParser(description="Generate wandb Report for GarmentGen training")
    parser.add_argument("--type", required=True, choices=["vae", "ldm"], help="Training type")
    parser.add_argument("--run-name", default=None, help="wandb run display name")
    parser.add_argument("--run-id", default=None, help="wandb run internal ID")
    parser.add_argument("--title", default=None, help="Report title")
    parser.add_argument("--entity", default=ENTITY_DEFAULT, help=f"wandb entity (default: {ENTITY_DEFAULT})")
    parser.add_argument("--project", default=PROJECT_DEFAULT, help=f"wandb project (default: {PROJECT_DEFAULT})")
    args = parser.parse_args()

    if not args.run_name and not args.run_id:
        parser.error("At least one of --run-name or --run-id is required")

    title = args.title or f"{args.type.upper()} Training Report"
    runset = build_runset(args.entity, args.project, args.run_name, args.run_id)

    if args.type == "vae":
        blocks = build_vae_blocks(runset)
    else:
        blocks = build_ldm_blocks(runset)

    report = wr.Report(
        project=args.project,
        entity=args.entity,
        title=title,
        width="fixed",
        blocks=[wr.H1(text=title)] + blocks,
    )
    report.save()
    print(f"Report created: {report.url}")


if __name__ == "__main__":
    main()
