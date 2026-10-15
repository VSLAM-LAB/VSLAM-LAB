"""
Module: VSLAM-LAB - Datasets - extra-files - stream_iphone.py
- Author: Alejandro Fontan Villacampa
- Assisted by: Claude (Fable 5.1)
- Version: 1.1
- Created: 2026-10-12
- Updated: 2026-10-12
- License: GPLv3 License

Streams an iPhone running the ARStream app (github.com/VSLAM-LAB/arstream) to any number of subscribers
so SLAM systems run online: RGB + intrinsics + ARKit pose, over USB (default, macOS) or Wi-Fi
(--wifi / --host), relayed over ZeroMQ PUB/SUB (default tcp://*:5556) as ARS1 messages (one per
frame, the phone's bytes untouched; {"event": "end"} when the relay stops). The relay (`arstream
publish`) runs in the arstream repository's own environment (cloned to Datasets/sources/arstream,
installed on first use); this script runs in the vslamlab environment and

  1. over USB on macOS, brings the app to the foreground on the phone first (see record_iphone.py),
  2. writes a VSLAM-LAB calibration.yaml from the first frame (default
     VSLAM-LAB-Benchmark/IPHONE/stream/calibration.yaml) before publishing it, so a baseline started
     afterwards can take it as calibration_yaml:,
  3. publishes until Ctrl-C or --duration.

With --run <baseline>, it also starts that baseline's online entry point (the pixi task
execute-<mode>-stream, command from BaselineVSLAMLAB.build_stream_command) once the calibration exists,
writing to VSLAM-LAB-Evaluation/stream_<baseline>/ (--exp-folder), and --view adds the rerun viewer as
another subscriber. Ctrl-C (or --duration) stops the relay first; its end-of-stream message makes the
baseline save its trajectory and exit, and this script returns once every subprocess is gone.

    pixi run stream-iphone                                   # relay only: USB -> tcp://*:5556
    pixi run stream-iphone --wifi --bind tcp://*:6000
    pixi run stream-iphone --run orbslam3-dev                # relay + ORB-SLAM3-DEV online (Ctrl-C ends both)
    pixi run stream-iphone --run orbslam3-dev --view --duration 30 --verbose 0
    pixi run view-iphone --sub                               # the rerun viewer on a running relay
"""

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from arstream_tools import ARSTREAM_PATH, add_phone_args, arstream_task, launch_app, phone_args  # noqa: E402
from Baselines.get_baseline import get_baseline  # noqa: E402
from path_constants import VSLAMLAB_BENCHMARK, VSLAMLAB_EVALUATION  # noqa: E402
from utilities import make_printers  # noqa: E402

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
print_info, print_warning = make_printers(SCRIPT_LABEL)

DEFAULT_BIND = "tcp://*:5556"
DEFAULT_CALIBRATION_YAML = VSLAMLAB_BENCHMARK / "IPHONE" / "stream" / "calibration.yaml"
CALIBRATION_TIMEOUT_S = 60.0   # phone connect + first frame
SHUTDOWN_TIMEOUT_S = 60.0      # a baseline's shutdown (bundle adjustment, trajectory save) after the end of stream


def subscriber_url(bind: str) -> str:
    """The URL subscribers on this machine connect to for a PUB endpoint (tcp://*:5556 -> tcp://localhost:5556)."""
    return bind.replace("//*:", "//localhost:").replace("//0.0.0.0:", "//localhost:")


def start(command: list[str] | str, name: str, cwd: Path | None = None) -> subprocess.Popen:
    """A subprocess in its own process group, so the terminal's Ctrl-C only reaches this script and the
    shutdown order below is ours to decide."""
    print_info(f"Starting {name}: {command if isinstance(command, str) else shlex.join(command)}")
    return subprocess.Popen(command, shell=isinstance(command, str), cwd=cwd, start_new_session=True)


def interrupt(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)


def wait(process: subprocess.Popen, name: str, timeout: float) -> None:
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print_warning(f"{name} did not exit {timeout:g} s after its stream ended - killing it")
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()


def has_pixi_task(environment: str, task: str) -> bool:
    """Whether VSLAM-LAB's pixi.toml defines `task` for `environment` (its execute-<mode>-stream forward)."""
    result = subprocess.run(["pixi", "task", "list", "--manifest-path", str(REPO_ROOT / "pixi.toml"), "-e", environment],
                            capture_output=True, text=True)
    return task in result.stdout.split()


def wait_for_calibration(path: Path, relay: subprocess.Popen) -> bool:
    deadline = time.monotonic() + CALIBRATION_TIMEOUT_S
    while time.monotonic() < deadline:
        if path.is_file():
            return True
        if relay.poll() is not None:
            return False
        time.sleep(0.2)
    return False


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # progress lines land in a log as they happen, not at exit
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_phone_args(parser)
    parser.add_argument("--bind", default=DEFAULT_BIND, help=f"ZeroMQ PUB endpoint (default {DEFAULT_BIND})")
    parser.add_argument("--calibration-yaml", type=Path, default=DEFAULT_CALIBRATION_YAML,
                        help=f"VSLAM-LAB calibration.yaml written from the first frame (default {DEFAULT_CALIBRATION_YAML})")
    parser.add_argument("--fps", type=float, default=30.0, help="nominal frame rate written to the calibration (default 30)")
    parser.add_argument("--duration", type=float, help="stop after this many seconds (default: until Ctrl-C)")
    parser.add_argument("--run", metavar="BASELINE", help="also run this baseline online (its execute-<mode>-stream task), e.g. orbslam3-dev")
    parser.add_argument("--mode", default="mono", help="mode of the baseline's stream entry point (default mono)")
    parser.add_argument("--exp-folder", type=Path, help="where --run writes its trajectory (default VSLAM-LAB-Evaluation/stream_<baseline>)")
    parser.add_argument("--verbose", type=int, choices=[0, 1], help="override the baseline's verbose parameter (its viewer)")
    parser.add_argument("--view", action="store_true", help="also show the rerun viewer as a subscriber")
    args = parser.parse_args()

    # Everything --run needs is checked before the phone is touched
    baseline_command = None
    if args.run:
        baseline = get_baseline(args.run)
        if not hasattr(baseline, "baseline_name"):
            sys.exit(f"{SCRIPT_LABEL}unknown baseline '{args.run}'")
        installed, message = baseline.is_installed()
        if not installed:
            sys.exit(f"{SCRIPT_LABEL}{args.run} is {message}: run 'pixi run install-baseline {args.run}' first")
        if not has_pixi_task(args.run, f"execute-{args.mode}-stream"):
            sys.exit(f"{SCRIPT_LABEL}{args.run} has no execute-{args.mode}-stream task in pixi.toml (no online {args.mode} entry point)")
        # mono runs keep the stream_<baseline> folder of before; other modes get their own (stream_<baseline>_rgbd)
        default_folder = f"stream_{args.run}" + ("" if args.mode == "mono" else f"_{args.mode}")
        exp_folder = (args.exp_folder or VSLAMLAB_EVALUATION / default_folder).resolve()
        exp_folder.mkdir(parents=True, exist_ok=True)
        overrides = {} if args.verbose is None else {"verbose": args.verbose}
        baseline_command = baseline.build_stream_command(args.mode, subscriber_url(args.bind), args.calibration_yaml.resolve(),
                                                         exp_folder, overrides=overrides)

    args.calibration_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_yaml.unlink(missing_ok=True)  # its appearance is the "first frame received" signal below
    print_info(f"Streaming the phone on {args.bind}; calibration -> {args.calibration_yaml}")
    launch_app(args)

    publish_args = [*phone_args(args), "--bind", args.bind, "--calibration-yaml", str(args.calibration_yaml), "--fps", str(args.fps)]
    if args.duration is not None:
        publish_args += ["--duration", str(args.duration)]
    relay_runner = arstream_task("publish")
    if not relay_runner.is_installed():
        relay_runner.install()
    relay_command = ["pixi", "run", "--manifest-path", str(relay_runner.manifest), "--frozen", "publish", *publish_args]

    if baseline_command is None and not args.view:
        # Relay only: the terminal's Ctrl-C reaches the relay itself (end-of-stream message, exit)
        try:
            relay_runner.run_args(publish_args)
        except (KeyboardInterrupt, subprocess.CalledProcessError):
            pass
        return

    relay = start(relay_command, "relay", cwd=ARSTREAM_PATH)
    subscribers: list[tuple[subprocess.Popen, str]] = []
    started = False
    try:
        if not wait_for_calibration(args.calibration_yaml, relay):
            if relay.poll() is not None:
                print_warning("The relay could not connect to the phone (see above) - nothing started")
            else:
                print_warning(f"No frame arrived from the phone within {CALIBRATION_TIMEOUT_S:g} s (is the ARStream app open?) - nothing started")
            return
        print_info(f"Calibration written: {args.calibration_yaml}")
        if args.mode == "rgbd" and "rgb+depth" not in args.calibration_yaml.read_text(encoding="utf-8"):
            print_warning("The phone streams no depth (not a LiDAR phone, or Depth off in the app): an rgbd run needs it "
                          "- use --mode mono")
            return
        started = True

        if args.view:
            subscribers.append((start(["pixi", "run", "--manifest-path", str(relay_runner.manifest), "--frozen", "view", "--sub",
                                       subscriber_url(args.bind)], "viewer", cwd=ARSTREAM_PATH), "viewer"))
        if baseline_command is not None:
            subscribers.append((start(baseline_command, args.run, cwd=REPO_ROOT), args.run))

        # Until the relay stops (--duration, phone disconnect) or a subscriber exits on its own
        while relay.poll() is None and all(process.poll() is None for process, _ in subscribers):
            time.sleep(0.5)
        for process, name in subscribers:
            if process.poll() is not None:
                print_info(f"{name} exited ({process.returncode})")
    except KeyboardInterrupt:
        print_info("Stopping the stream ...")
    finally:
        # Relay first: its end-of-stream message is what makes the subscribers finish and save. A subscriber that
        # is not reading the stream right now (a baseline paused from its viewer) never sees it, so each one also
        # gets the same Ctrl-C the terminal would have given it: their handlers stop the subscription (and lift a
        # pause), after which they save and exit like on the end message.
        interrupt(relay)
        wait(relay, "relay", SHUTDOWN_TIMEOUT_S)
        for process, name in subscribers:
            interrupt(process)
            wait(process, name, SHUTDOWN_TIMEOUT_S)
            print_info(f"{name} finished ({process.returncode})")
        if baseline_command is not None and started:
            print_info(f"Results in {exp_folder}")


if __name__ == "__main__":
    main()
