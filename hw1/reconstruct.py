"""Thin CLI over utils.py (reconstruct.py).

    Full module guide migrated to docs/reconstruct.md — baseline +
    selected runs, status-driven selection, write-back, and CLI flags.
"""
import os
import re
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import api                                              # noqa: E402
import utils                                            # noqa: E402


# Batch names are floor-qualified (`floor1_baseline`); capture directories are not.
_FLOOR_PREFIX = re.compile(r"^floor\d+_")


def _capture_dir_name(batch_name):
    """The capture-dir basename a batch name ends with: floor1_baseline -> baseline."""
    return _FLOOR_PREFIX.sub("", batch_name)


def _segments_from_frames(frames):
    """Sorted frame indices -> maximal contiguous segments, including singletons."""
    segments = []
    for frame in sorted(set(int(value) for value in frames)):
        if segments and frame == segments[-1][-1] + 1:
            segments[-1].append(frame)
        else:
            segments.append([frame])
    return segments


def _plan_query_selected(exp, query_path):
    """SPARQL file -> (frames, segments) to reconstruct.

    Runs the query against the experiment graph via
    api.frames_from_sparql_query, cuts the flat frame list into maximal
    contiguous segments so no hidden temporal jump is reconstructed.
    Returns (None, []) when the selection is empty (nothing to run).
    """
    try:
        selected = api.frames_from_sparql_query(exp, query_path)
    except OSError as exc:
        raise ValueError(f"cannot read query {query_path!r}: {exc}") from None
    segments = _segments_from_frames(selected)
    if not segments:
        print("[reconstruct] SPARQL selection is EMPTY: no selected run.")
        return None, []
    frames = [frame for segment in segments for frame in segment]
    spans = ", ".join(f"{segment[0]}..{segment[-1]}({len(segment)})"
                      for segment in segments[:8])
    if len(segments) > 8:
        spans += f", … +{len(segments) - 8} more"
    print(f"[reconstruct] SPARQL selection: {len(segments)} segment(s), "
          f"{len(frames)} frames [{spans}]")
    return frames, segments



def _segment_metadata(segments):
    """Statusless evidence for the temporal discontinuities in a selected run."""
    if not segments:
        return {"spliceCount": 0, "maxGapLength": 0}
    gaps = [max(0, int(right[0]) - int(left[-1]) - 1)
            for left, right in zip(segments, segments[1:])]
    return {"spliceCount": max(0, len(segments) - 1),
            "maxGapLength": max(gaps, default=0)}


def main():
    """Workflow: declare -> measure -> select -> reconstruct -> show.

    1. ``api.py declare`` scaffolds the experiment (factor selection).
    2. ``api.py experiment`` measures the raw data against those factors.
    3. Here, frame selection is one of two modes:
       fullbatch (frames=None, the whole batch) or a SPARQL file
       (``--sparql-query``) whose ?frame/?frameIndex bindings become the
       frame list fed to ``utils.reconstruct``.
    4. One reconstruction run, one Mean L2 number, one Open3D window.
    """
    parser = argparse.ArgumentParser(
        description="Reconstruct a capture (fullbatch or SPARQL-selected frames), "
                    "print Mean L2, show the cloud + trajectories.")
    # dest default is None ON PURPOSE: it is the only way to tell "caller asked for
    # open3d" from "caller said nothing", and the icpBackend rule below needs that.
    parser.add_argument('-v', '--version', type=str, default=None,
                        help='open3d or my_icp (default: the experiment\'s icpBackend '
                             'setting, else open3d). Passing this AND an experiment '
                             'that records a different backend is an error.')
    parser.add_argument('--data_root', type=str,
                        default=os.path.join("eval", "_data", "second_floor",
                                             "baseline"),
                        help='capture dir to reconstruct (rgb/ depth/ GT_pose.npy). '
                             'Stays explicit even though the experiment names the '
                             'same directory in hw1:batchFile.')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                        help='registration voxel size in metres (default: 0.05)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Open3D RANSAC seed (default: 0)')
    parser.add_argument('--no-vis', action='store_true',
                        help='skip the Open3D window (print metrics only)')
    parser.add_argument('--experiment', type=str, default=None,
                        help='assessed experiment Turtle (experiments/<expname>.ttl): '
                             'already measured by `api.py experiment`. The run value '
                             'is written back below the machine marker. Omit for a '
                             'plain whole-batch visual run.')
    parser.add_argument('--selection-query', metavar='QUERY.rq', default=None,
                        help='SPARQL SELECT over --experiment binding ?frame (frame IRI) '
                             'or ?frameIndex (integer). The bindings become the frame '
                             'list; results are cut into contiguous segments.')
    parser.add_argument('--sparql-query', dest='selection_query', metavar='QUERY.rq', default=None,
                        help='alias of --selection-query.')
    parser.add_argument('--no-write', action='store_true',
                        help='dry run: print the results but write nothing into the '
                             'experiment file')
    args = parser.parse_args()

    if args.voxel_size <= 0:
        parser.error("--voxel-size must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.experiment is None and args.selection_query is not None:
        parser.error("--sparql-query requires --experiment: it queries that experiment's "
                     "local RDF graph.")

    data_root = args.data_root
    version = args.version

    exp = None
    if args.experiment is not None:
        exp = api.read_experiment(args.experiment)
        print(f"[reconstruct] {args.experiment}: experiment {exp['exp_name']} on batch "
              f"{exp['batch_name']!r}, {len(exp['frame_status'])} frames, "
              f"{len(exp['pair_status'])} pairs")

        # Guard: is this experiment even about this capture?
        exp_dir_name = _capture_dir_name(exp["batch_name"])
        arg_dir_name = os.path.basename(os.path.normpath(data_root))
        if exp_dir_name != arg_dir_name:
            print(f"[reconstruct] *** WARNING: BATCH MISMATCH ***\n"
                  f"[reconstruct]   experiment {args.experiment} is on batch "
                  f"{exp['batch_name']!r} (capture dir {exp_dir_name!r})\n"
                  f"[reconstruct]   --data_root is {data_root!r} "
                  f"(capture dir {arg_dir_name!r})\n"
                  f"[reconstruct]   the score of one capture would be written into "
                  f"another capture's experiment.", file=sys.stderr)

        # The RECORDED icpBackend setting is authoritative.
        set_backend = exp["settings"].get("icpBackend")
        if set_backend is not None:
            set_backend = str(set_backend)
            if version is None:
                version = set_backend
                print(f"[reconstruct] icpBackend from experiment: {version}")
            elif version != set_backend:
                parser.error(
                    f"--version {version!r} disagrees with the icpBackend setting "
                    f"{set_backend!r} recorded in {args.experiment}. That setting is "
                    f"part of the treatment this experiment declares; write a new "
                    f"declaration for the other backend instead of overriding it.")

    if version is None:
        version = 'open3d'

    # Selection: fullbatch (whole batch) or SPARQL (frame list from query).
    if args.selection_query is not None:
        frames, segments = _plan_query_selected(exp, args.selection_query)
        if frames is None:
            return  # empty selection: nothing to reconstruct, nothing written
        mode = "selected"
    else:
        frames, segments = None, []
        mode = "baseline"

    # The map is only ever looked at by the visualiser; skipping it halves peak
    # memory and utils.reconstruct guarantees the trajectory is identical either way.
    build_cloud = not args.no_vis

    prior_measurements = []
    prior_callback = None
    measure_prior = (
        exp is not None and not args.no_write and mode == "baseline" and
        "PriorWarpDepthResidual" in exp["selected"])
    if measure_prior:
        prior_callback = api.make_prior_warp_measurement_callback(
            data_root,
            exp["settings"].get("priorWarpDepthGate", 0.10),
            prior_measurements)

    print(f"[reconstruct] === {mode} run ===")
    pcd, pred_cam_pos = utils.reconstruct(
        data_root, version, frames=frames, build_cloud=build_cloud,
        voxel_size=args.voxel_size, registration_seed=args.seed,
        prior_transform_callback=prior_callback)
    l2 = utils.visualize_and_evaluate(
        pcd, pred_cam_pos, data_root, frames=frames,
        title=f'{os.path.basename(os.path.normpath(data_root))} '
              f'({version}, {mode})',
        show=not args.no_vis)

    if exp is None:
        print(f"[reconstruct] no --experiment: Mean L2 Distance = {l2!r} not written "
              f"anywhere")
    elif args.no_write:
        print(f"[reconstruct] --no-write: {mode} Mean L2 Distance = {l2!r} NOT written "
              f"to {args.experiment}")
    else:
        if measure_prior:
            written = api.write_pair_measurements(
                args.experiment, "PriorWarpDepthResidual", prior_measurements)
            print(f"[reconstruct] wrote PriorWarpDepthResidual for "
                  f"{written}/{len(prior_measurements)} reconstructed pair(s) "
                  f"-> {args.experiment}")
        frame_count = len(pred_cam_pos)
        if frames is not None and frame_count != len(frames):
            print(f"[reconstruct] *** WARNING: {mode} run selected {len(frames)} "
                  f"frames but reconstructed {frame_count}; some frame failed to "
                  f"load. hw1:runFrameCount reports what was consumed.",
                  file=sys.stderr)
        run_metadata = {}
        if mode == "selected":
            run_metadata.update(_segment_metadata(segments))
        api.write_run(
            args.experiment, mode, {"mapMeanL2": l2},
            used_frames=frames if frames is not None else None,
            frame_count=frame_count, metadata=run_metadata)
        print(f"[reconstruct] wrote {mode} run: Mean L2 Distance = {l2!r} "
              f"(stored as hw1:mapMeanL2 at file precision), "
              f"runFrameCount = {frame_count} -> {args.experiment}")


if __name__ == '__main__':
    main()
