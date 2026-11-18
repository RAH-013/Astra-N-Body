import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_frames(path):
    frames = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"frame", "step", "particle_id", "x", "y", "z", "mass"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV does not contain the required trajectory columns.")
        for row in reader:
            frame_id = int(row["frame"])
            frames.setdefault(frame_id, []).append({
                "step": int(row["step"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "mass": float(row["mass"]),
            })
    if not frames:
        raise ValueError("The CSV contains no particle records.")
    return dict(sorted(frames.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()

    frames = load_frames(args.csv)
    args.output.mkdir(parents=True, exist_ok=True)
    frame_ids = list(frames)
    last = frames[frame_ids[-1]]

    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        [item["x"] for item in last],
        [item["y"] for item in last],
        [item["z"] for item in last],
        c=[item["mass"] for item in last],
        s=8,
        alpha=0.8,
    )
    axis.set_title(f"ASTRA N-BODY — step {last[0]['step']}")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    fig.colorbar(scatter, ax=axis, label="Mass")
    fig.tight_layout()
    fig.savefig(args.output / "astra_snapshot.png", dpi=200)
    plt.close(fig)

    if not args.no_gif and len(frame_ids) > 1:
        fig = plt.figure(figsize=(9, 8))
        axis = fig.add_subplot(111, projection="3d")
        all_particles = [item for frame in frames.values() for item in frame]
        extent = max(
            max(abs(item["x"]) for item in all_particles),
            max(abs(item["y"]) for item in all_particles),
            max(abs(item["z"]) for item in all_particles),
            1.0,
        )
        axis.set_xlim(-extent, extent)
        axis.set_ylim(-extent, extent)
        axis.set_zlim(-extent, extent)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        points = axis.scatter([], [], [], s=8, alpha=0.8)
        title = axis.set_title("ASTRA N-BODY")

        def update(frame_id):
            particles = frames[frame_id]
            points._offsets3d = (
                [item["x"] for item in particles],
                [item["y"] for item in particles],
                [item["z"] for item in particles],
            )
            points.set_array([item["mass"] for item in particles])
            title.set_text(f"ASTRA N-BODY — step {particles[0]['step']}")
            return points, title

        animation = FuncAnimation(fig, update, frames=frame_ids, interval=args.interval, blit=False)
        animation.save(args.output / "astra_evolution.gif", writer=PillowWriter(fps=max(1, round(1000 / args.interval))))
        plt.close(fig)

    print(f"Snapshot: {(args.output / 'astra_snapshot.png').resolve()}")
    if not args.no_gif and len(frame_ids) > 1:
        print(f"Animation: {(args.output / 'astra_evolution.gif').resolve()}")


if __name__ == "__main__":
    main()
