import os
import torch
import numpy as np
import tensorrt as trt
import logging
import cvcuda


# NOTE: One must import PyCuda driver first, before CVCUDA or VPF otherwise
# things may throw unexpected errors.
import pycuda.driver as cuda  # noqa: F401

import os
import sys
import json
import logging
from datetime import datetime
import argparse
import subprocess
from collections import deque
import cvcuda
import torch
import nvtx
import pandas


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(name)s:%(lineno)d] %(asctime)s %(levelname)-6s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class CvCudaPerf:
    """
    This class helps keep track of the CPU and GPU run-time performance of
    arbitrary python code spinets. This class do not have the functionality to
    actually perform any benchmarking since that is already done by NVIDIA NSYS.
    This must be used in conjunction with the script like benchmark.py to actually
    get the performance numbers.

    This class acts as an extension of the NVTX markers API. It allows us to capture
    application specific ranges with a few extra meta-data such as what range was
    inside a batch and what was the batch size, etc.
    """

    def __init__(
        self,
        obj_name,
        default_args,
    ):
        """
        Initializes a new instance of the `perf_utils` class.
        :param obj_name: The name of the object used for performance benchmarking.
        :param default_args: The usual set of command line arguments used to launch the CV-CUDA sample.
        """
        self.obj_name = obj_name
        self.command_line_args = default_args

        if hasattr(self.command_line_args, "device_id"):
            self.device_id = self.command_line_args.device_id
        else:
            raise ValueError("device_id must be provided in the default_args.")

        if hasattr(self.command_line_args, "output_dir"):
            self.output_dir = self.command_line_args.output_dir
        else:
            raise ValueError("output_dir must be provided in the default_args.")

        self.logger = logging.getLogger(__name__)
        # We will use a stack to record the push/pop range operations.
        self.stack = deque()
        self.stack_path = self.obj_name
        # We will maintain 3 different dictionaries to store the data.
        # 1. timing_info: to store CPU and GPU timings of NVTX ranges.
        # 2. batch_info: to store batch size and batch index of NVTX ranges.
        # 3. inside_batch_info: to store which NVTX ranges are part of a batch.
        self.timing_info = {}
        self.batch_info = {}
        self.inside_batch_info = []
        self.deleted_range_info = []
        self.is_inside_batch = 0
        self.total_batches_processed = {}
        # Check if the benchmark.py script was used to run this. We do so
        # by checking whether an environment variable only set by that script is
        # present or not.
        if os.environ.get("BENCHMARK_PY"):
            self.should_benchmark = True
            self.logger.info("Benchmarking mode is turned on.")
        else:
            self.should_benchmark = False
            self.logger.warning(
                "perf_utils is used without benchmark.py. "
                "Benchmarking mode is turned off."
            )
        self.logger.info("Using CV-CUDA version: %s" % cvcuda.__version__)

    def push_range(
        self, message=None, color="blue", domain=None, category=None, batch_idx=None
    ):
        """
        Pushes a code range on to the stack for performance benchmarking.
        :param message: A message associated with the annotated code range.
        :param color: A color associated with the annotated code range.
        :param domain: Name of a domain under which the code range is scoped.
        :param category: A string or an integer specifying the category within the domain
        under which the code range is scoped. If unspecified, the code range
        is not associated with a category.
        :param batch_idx: If this range is associated with a batch, then its batch number.
         All the ranges pushed after this will be automatically associated with this batch.
        """
        if batch_idx is not None:
            message += "_%d" % batch_idx
            self.is_inside_batch += 1

        nvtx.push_range(message, color, domain, category)

        if self.should_benchmark:
            self.stack.append((message, batch_idx))
            self.stack_path = os.path.join(self.stack_path, message)

    def pop_range(self, domain=None, total_items=None, delete_range=False):
        """
        Pops a code range off of the stack for performance benchmarking.
        :param domain: Name of a domain under which the code range is scoped.
        :param total_items: The number of items processed in this range.
        :param delete_range: Flag specifying whether the range should be completely deleted
         instead of just popping it out. This will remove all traces of this range from
         the benchmarks. Useful if the code being benchmarked fails and one wants to
         remove its range in that case.
        """
        if self.should_benchmark:
            # Grab the message and optional batch index from the stack.
            message, batch_idx = self.stack.pop()

            if not delete_range:
                # Add only if this range was not meant for deletion.
                self.timing_info[self.stack_path] = (
                    0,
                    0,
                )  # Placeholders for CPU and GPU times respectively.
                # Actual timing information will be recorded and pulled from NSYS by a
                # script like benchmark.py.
            else:
                # This range was meant for deletion. We did not add it to the timing_info
                # but all the previously added children of this range must also be deleted.
                # We will do that later in the finalize to avoid costing us time here.
                # For that, we will save this stack path so that we can remove all the
                # orphan nodes later.
                self.deleted_range_info.append(self.stack_path)

            if self.is_inside_batch > 0 and not delete_range:
                self.inside_batch_info.append(self.stack_path)

            # Record the batch information if it was present.
            if total_items is not None:
                if self.is_inside_batch <= 0:
                    raise ValueError(
                        "Non zero value for total_items in pop_range can only be "
                        "passed once inside a batch. No known batch was pushed previously. Please "
                        "push a batch first by using the batch_idx in the push_range()."
                    )

                self.is_inside_batch -= 1  # Decrement this by one.

                if not delete_range:
                    # Add to batch info only if this range was not meant for deletion.
                    self.batch_info[self.stack_path] = (batch_idx, total_items)

                    # Maintain a count of the number of items processed in various batches.
                    if total_items > 0:
                        batch_level_prefix = os.path.dirname(self.stack_path)

                        if batch_level_prefix not in self.total_batches_processed:
                            self.total_batches_processed[batch_level_prefix] = 0
                        self.total_batches_processed[batch_level_prefix] += 1

            # Unwind the stack to point to the previous path(i.e. directory like expression)
            # e.g. one level above.
            self.stack_path = os.path.dirname(self.stack_path)

        nvtx.pop_range(domain)

    def finalize(self):
        """
        Finalizes the performance benchmark data dictionary and saves it in the output folder
        as a JSON file. The benchmark data will be all zeros at this point. Actual data is
        captured and pulled from the NSYS reports when benchmark.py is run after this.
        """
        if self.should_benchmark:
            if len(self.stack):
                raise Exception(
                    "Unable to finalize timing info. The stack was non empty with %d"
                    " item(s) still not popped." % len(self.stack)
                )

            # Remove the keys from the timing_info which starts with any key in the
            # deleted_range_info. That makes sure that we not only delete the current
            # key but also all of its previous children which were added but not deleted.
            timing_info_keys = list(self.timing_info.keys())
            for key_delete in self.deleted_range_info:
                for k in timing_info_keys:
                    if k.startswith(key_delete):
                        self.timing_info.pop(k, None)

            # Build a dictionary containing the timing information and some metadata
            # about this run.
            # The overall structure of this would be:
            # {
            #   "data" : {
            #       ...
            #   }
            #   "data_stats_minus_warmup" : {
            #       ...
            #   }
            #   "gpu_metrics" : {
            #       ...
            #   },
            #   "batch_info" : {
            #       ...
            #   }
            #   "inside_batch_info" : [
            #       ...
            #   ]
            #   "meta" : {
            #       ...
            #   }
            # }
            #
            # The data field stores timing info of all batches keyed with raw flattened
            # names of the NVTX push/pop ranges.
            # The data_stats_minus_warmup stores various stats (e.g. mean, median) for
            # NVTX ranges across all the batches.
            # The gpu_metrics field stores various GPU metrics (e.g. power and utilization)
            # The batch_info stores the batch index and batch size of each batch.
            # The inside batch info is list of NVTX range names which executed inside
            # a batch.
            # The meta field stores various metadata about this run.
            #
            # NOTE: The numbers in the data/mean_batch_data field are all zero.
            #  i.e. They are only acting as placeholders. The actual numbers will be
            #       captured and pulled from NSYS when benchmarking is run with
            #       the benchmark.py script.
            #
            benchmark_dict = {
                "data": self.timing_info,
                "data_stats_minus_warmup": {},
                "gpu_metrics": {},
                "batch_info": self.batch_info,
                "inside_batch_info": self.inside_batch_info,
                "meta": {},
            }

            # Then we add basic details about this run and its configuration as meta.
            benchmark_dict["meta"] = {
                "obj_name": self.obj_name,
                "measurement_unit": "milliseconds",
                "total_batches": self.total_batches_processed,
                "cvcuda_version": cvcuda.__version__,
                "pytorch_version": torch.__version__,
                "python_version": sys.version,
            }
            if torch.cuda.device_count():
                benchmark_dict["meta"]["device"] = {
                    "id": self.device_id,
                    "name": torch.cuda.get_device_name(self.device_id),
                }
            else:
                benchmark_dict["meta"]["device"] = {
                    "id": self.device_id,
                    "name": "CPU",
                }

            benchmark_dict["meta"]["args"] = {}
            if self.command_line_args:
                for arg in vars(self.command_line_args):
                    benchmark_dict["meta"]["args"][arg] = getattr(
                        self.command_line_args, arg
                    )

            # The benchmark_dict is ready at this point. Convert it to JSON and write it.
            benchmark_json = json.dumps(benchmark_dict, indent=4)
            benchmark_file_path = os.path.join(self.output_dir, "benchmark.json")
            with open(benchmark_file_path, "w") as f:
                f.write(benchmark_json)
            self.logger.info(
                "Placeholder benchmark.json was written to: %s" % benchmark_file_path
            )

            return benchmark_dict

        else:
            return {}


def maximize_clocks(logger, device_id=0):
    """
    Maximizes the GPU clocks. Useful to do it before any type of performance
    benchmarking.
    :param device_id: The GPU device ID whose clocks should be maximized.
    """
    logger.info("Trying to maximize the GPU clocks for device: %d" % device_id)

    gpu_available = torch.cuda.device_count() > 0

    was_persistence_mode_on = False
    current_power_limit = None

    if not gpu_available:
        logger.warning("No GPUs available to maximize the clocks.")
        return False, was_persistence_mode_on, current_power_limit

    # 1. Enable persistence mode if not already done.
    proc_ret = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=persistence_mode",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.warning("Unable to query persistence mode.")
        was_persistence_mode_on = False
    else:
        was_persistence_mode_on = proc_ret.stdout.decode() == "Enabled"

    if not was_persistence_mode_on:
        proc_ret = subprocess.run(
            ["nvidia-smi", "--persistence-mode=1"], stdout=subprocess.PIPE
        )
        if proc_ret.returncode:
            logger.error("Unable to set persistence mode.")
            return False, was_persistence_mode_on, current_power_limit
        else:
            logger.info("Turned on persistence mode.")

    # 2. Disable auto boost before locking clocks.
    proc_ret = subprocess.run(
        ["nvidia-smi", "--auto-boost-default=DISABLED"], stdout=subprocess.PIPE
    )
    if proc_ret.returncode:
        logger.warning("Unable to turn off auto boost mode.")
    else:
        logger.info("Turned off auto-boost mode.")

    # 3. Maximize the power limits.
    # Get the current value first and save it.
    proc_ret = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=power.limit",
            "--format=csv,nounits,noheader",
            "-i=%d" % device_id,
        ],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.warning("Unable to query the current power limit.")
    else:
        current_power_limit = float(proc_ret.stdout.decode())

        # Get the maximum value after that.
        proc_ret = subprocess.run(
            [
                "nvidia-smi",
                "-i=%d" % device_id,
                "--query-gpu=power.max_limit",
                "--format=csv,nounits,noheader",
            ],
            stdout=subprocess.PIPE,
        )
        if proc_ret.returncode:
            logger.warning("Unable to query maximum power limit.")
        else:
            # Set the limit.
            max_power_limit = float(proc_ret.stdout.decode())
            proc_ret = subprocess.run(
                [
                    "nvidia-smi",
                    "-i=%d" % device_id,
                    "--power-limit=%f" % max_power_limit,
                ],
                stdout=subprocess.PIPE,
            )

            if proc_ret.returncode:
                logger.warning("Unable to set maximum power limit.")
            else:
                logger.info("Set the maximum power limit to %f." % max_power_limit)

    # 4. Query the maximum allowed Graphics clock and lock it.
    proc_ret = subprocess.run(
        [
            "nvidia-smi",
            "-i=%d" % device_id,
            "--query-gpu=clocks.max.graphics",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.error(
            "Unable to query the maximum graphics clock. Clocks were not maximized."
        )
        return False, was_persistence_mode_on, current_power_limit
    else:
        max_graphics_clock = float(proc_ret.stdout.decode())
        proc_ret = subprocess.run(
            [
                "nvidia-smi",
                "-i=%d" % device_id,
                "--lock-gpu-clocks=%d,%d" % (max_graphics_clock, max_graphics_clock),
            ],
            stdout=subprocess.PIPE,
        )
        if proc_ret.returncode:
            logger.error("Unable to lock the GPU clock. Clocks were not maximized.")
            return False, was_persistence_mode_on, current_power_limit
        else:
            logger.info("Locked the GPU clock to %d." % (max_graphics_clock))

    # 5. Query the maximum allowed memory clock and lock it.
    proc_ret = subprocess.run(
        [
            "nvidia-smi",
            "-i=%d" % device_id,
            "--query-gpu=clocks.max.memory",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.error(
            "Unable to query the maximum memory clock. Clocks were not maximized."
        )
        return False, was_persistence_mode_on, current_power_limit
    else:
        max_memory_clock = float(proc_ret.stdout.decode())
        proc_ret = subprocess.run(
            [
                "nvidia-smi",
                "-i=%d" % device_id,
                "--lock-memory-clocks=%d,%d" % (max_memory_clock, max_memory_clock),
            ],
            stdout=subprocess.PIPE,
        )
        if proc_ret.returncode:
            logger.warning(
                "Unable to lock the memory clock. Clocks were not maximized."
            )
        else:
            logger.info("Locked the memory clock to %d." % max_memory_clock)

    # 6. Lock the application clocks. Specifies <memory,graphics> clocks as a pair
    proc_ret = subprocess.run(
        [
            "nvidia-smi",
            "-i=%d" % device_id,
            "--applications-clocks=%d,%d" % (max_memory_clock, max_graphics_clock),
        ],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.warning("Unable to lock the application clocks.")
    else:
        logger.info(
            "Locked the application clocks to %d, %d."
            % (max_memory_clock, max_graphics_clock)
        )

    # 7. Get the GPU Performance State. P0 state means the most performance.
    proc_ret = subprocess.run(
        [
            "nvidia-smi",
            "-i=%d" % device_id,
            "--query-gpu=pstate",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.warning("Unable to query the performance state of the GPU.")

        return False, was_persistence_mode_on, current_power_limit
    else:
        gpu_perf_state = proc_ret.stdout.decode().strip()
        logger.info("Current GPU performance state is %s." % gpu_perf_state)

        if gpu_perf_state == "P0":
            logger.info("Clocks for device %d are now maximized." % device_id)
            return True, was_persistence_mode_on, current_power_limit
        else:
            logger.info(
                "Unable to maximize GPU clocks of device %d to reach the P0 state."
                % device_id
            )
            return False, was_persistence_mode_on, current_power_limit


def reset_clocks(
    logger,
    device_id=0,
    was_persistence_mode_on=False,
    current_power_limit=None,
):
    """
    Resets the GPU clocks.
    """
    logger.info("Trying to reset the GPU clocks for device: %d" % device_id)

    gpu_available = torch.cuda.device_count() > 0

    if not gpu_available:
        logger.warning("No GPUs available to reset the clocks.")

    # 1. Reset the memory clock.
    proc_ret = subprocess.run(
        ["nvidia-smi", "-i=%d" % device_id, "--reset-memory-clocks"],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.warning("Unable to reset the memory clock back to normal.")
    else:
        logger.info("Reset the memory clock back to normal.")

    # 2. Reset GPU clock.
    proc_ret = subprocess.run(
        ["nvidia-smi", "-i=%d" % device_id, "--reset-gpu-clocks"],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.warning("Unable to reset the GPU clock back to normal.")
    else:
        logger.info("Reset the graphics clock back to normal.")

    # 3. Reset application clocks.
    proc_ret = subprocess.run(
        ["nvidia-smi", "-i=%d" % device_id, "--reset-applications-clocks"],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.warning("Unable to reset the application clocks back to normal.")
    else:
        logger.info("Reset the application clocks back to normal.")

    # 4. Reset the power limit.
    if current_power_limit:
        proc_ret = subprocess.run(
            [
                "nvidia-smi",
                "-i=%d" % device_id,
                "--power-limit=%f" % current_power_limit,
            ],
            stdout=subprocess.PIPE,
        )

        if proc_ret.returncode:
            logger.warning("Unable to reset the power limit back to normal.")
        else:
            logger.info("Reset the power limit to normal.")

    # 5. Enable auto boost back..
    proc_ret = subprocess.run(
        ["nvidia-smi", "--auto-boost-default=ENABLED"], stdout=subprocess.PIPE
    )
    if proc_ret.returncode:
        logger.warning("Unable to turn the auto boost mode back on.")
    else:
        logger.info("Turned the auto-boost mode back on.")

    # 6. Turn off persistence mode if it was enabled by us.
    if not was_persistence_mode_on:
        proc_ret = subprocess.run(
            ["nvidia-smi", "--persistence-mode=0"], stdout=subprocess.PIPE
        )
        if proc_ret.returncode:
            logger.warning("Unable to turn off the persistence mode.")
        else:
            logger.info("Turned off the persistence mode.")

    # 7. Get GPU Performance State
    proc_ret = subprocess.run(
        [
            "nvidia-smi",
            "-i=%d" % device_id,
            "--query-gpu=pstate",
            "--format=csv,nounits,noheader",
        ],
        stdout=subprocess.PIPE,
    )
    if proc_ret.returncode:
        logger.warning("Unable to query the performance state of the GPU.")

        return False, was_persistence_mode_on, current_power_limit
    else:
        gpu_perf_state = proc_ret.stdout.decode().strip()
        logger.info("Current GPU performance state is %s." % gpu_perf_state)


def get_default_arg_parser(
    message,
    supports_video=True,
    input_path=None,
    output_dir="/tmp",
    target_img_height=224,
    target_img_width=224,
    batch_size=4,
    device_id=0,
    supported_backends=["tensorrt", "pytorch"],
    backend="tensorrt",
    log_level="info",
    parser_type="vision",
):
    """
    Prepares and returns an argparse command line argument parser for the scripts
    that supports auto-benchmarking. This parser guarantees that all the scripts which can be
    benchmarked supports a basic set of command line arguments which allows us to run
    them in a uniform and consistent fashion.
    """

    # Check what kind of parser the user needs.
    # 1. A vision parser:
    #       Adds all of the most commonly used command-line arguments for a typical
    #       computer vision pipeline.
    # 2. A minimal parser:
    #       Only adds the arguments needed for performance benchmarking.
    #
    if parser_type not in ["vision", "minimal"]:
        raise ValueError("parser_type must either be 'vision' or 'minimal.")

    assets_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "assets",
    )

    assets_dir = "./"
    if not input_path:
        input_path = os.path.join(assets_dir, "source", "1.jpg")

    parser = argparse.ArgumentParser(
        message,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    if parser_type == "vision":
        if not supports_video:
            parser.add_argument(
                "-i",
                "--input_path",
                default=input_path,
                type=str,
                help="The path to a JPEG image or a directory containing JPG images "
                "to use as input. When pointing to a directory, only *.jpg images will be read.",
            )
        else:
            parser.add_argument(
                "-i",
                "--input_path",
                default=input_path,
                type=str,
                help="Either a path to a JPEG image/MP4 video or a directory containing JPG images "
                "to use as input. When pointing to a directory, only *.jpg images will be read.",
            )

    if parser_type in ["vision", "minimal"]:
        parser.add_argument(
            "-o",
            "--output_dir",
            default=output_dir,
            type=str,
            help="The folder where the output results should be stored.",
        )

    if parser_type == "vision":
        parser.add_argument(
            "-th",
            "--target_img_height",
            default=target_img_height,
            type=int,
            help="The height to which you want to resize the input_image before "
            "running inference.",
        )

        parser.add_argument(
            "-tw",
            "--target_img_width",
            default=target_img_width,
            type=int,
            help="The width to which you want to resize the input_image before "
            "running inference.",
        )

        parser.add_argument(
            "-b",
            "--batch_size",
            default=batch_size,
            type=int,
            help="The batch size.",
        )

    if parser_type in ["vision", "minimal"]:
        parser.add_argument(
            "-d",
            "--device_id",
            default=device_id,
            type=int,
            help="The GPU device to use for this sample.",
        )

    if parser_type == "vision":
        parser.add_argument(
            "-bk",
            "--backend",
            type=str,
            choices=supported_backends,
            default=backend,
            help="The inference backend to use. Currently supports %s."
            % ", ".join(supported_backends),
        )

    if parser_type in ["vision", "minimal"]:
        parser.add_argument(
            "-ll",
            "--log_level",
            type=str,
            choices=["info", "error", "debug", "warning"],
            default=log_level,
            help="Sets the desired logging level. Affects the std-out printed by the "
            "sample when it is run.",
        )

    return parser


def parse_validate_default_args(parser):
    """
    Parses and validates the values of the default command line arguments.
    """
    args = parser.parse_args()

    if hasattr(args, "input_path"):
        if not os.path.isdir(args.input_path) and not os.path.isfile(args.input_path):
            raise ValueError(
                "input_path is neither a valid file not a directory: %s"
                % args.input_path
            )

    if hasattr(args, "output_dir"):
        if not os.path.isdir(args.output_dir):
            raise ValueError(
                "output_dir is not a valid directory: %s" % args.output_dir
            )

    if hasattr(args, "batch_size"):
        if args.batch_size <= 0:
            raise ValueError("batch_size must be a value >=1.")

    if hasattr(args, "device_id"):
        if torch.cuda.device_count():
            if args.device_id < 0 or args.device_id >= torch.cuda.device_count():
                raise ValueError(
                    "device_id must be a valid value from 0 to %d."
                    % (torch.cuda.device_count() - 1)
                )

    if hasattr(args, "target_img_height"):
        if args.target_img_height < 10:
            raise ValueError("target_img_height must be a value >=10.")

    if hasattr(args, "target_img_width"):
        if args.target_img_width < 10:
            raise ValueError("target_img_width must be a value >=10.")

    return args


def summarize_runs(
    baseline_run_root,
    baseline_run_name="baseline",
    compare_run_roots=[],
    compare_run_names=[],
):
    """
    Summarizes one or more benchmark runs and prepares a pandas table showing the run per sample run-time
     and speed-up numbers.
    :param baseline_run_root: Folder containing one sub-folder per sample in which the benchmark.py
     styled JSON of the baseline run is stored.
    :param baseline_run_name: The display name of the column representing the first run in the table.
    :param compare_run_roots: Optional. A list of folder containing one sub-folder per sample in which the
     benchmark.py styled JSON of the other runs are stored. These runs are compared with the baseline run.
    :param compare_run_names: A list of display names of the column representing the comparison runs
     in the table. This must be of the same length as the `compare_run_json_paths`.
    :returns: A pandas table with the sample's name and its run time from the baseline run.
     If compare runs are given, it also returns their run times and the speed-up
     compared to the baseline run. The speedup is simply the run time of the sample from the compare run
     divided by its run time from the baseline run. If an sample's run time or speedup factor is not
     available, it simply puts "N/A".
    """

    def _parse_json_for_time(json_data):
        mean_all_procs = json_data["data_mean_all_procs"]
        sample_name_key = list(mean_all_procs.keys())[0]

        cpu_time_minus_warmup_per_item = mean_all_procs[sample_name_key]["run_sample"][
            "pipeline"
        ]["cpu_time_minus_warmup_per_item"]["mean"]

        return cpu_time_minus_warmup_per_item

    baseline_perf = {}
    if os.path.isdir(baseline_run_root):
        for path in os.listdir(baseline_run_root):
            if os.path.isdir(os.path.join(baseline_run_root, path)):
                json_path = os.path.join(baseline_run_root, path, "benchmark_mean.json")
                if os.path.isfile(json_path):
                    with open(json_path, "r") as f:
                        json_data = json.loads(
                            f.read()
                        )  # Storing by the name of the sample

                        baseline_perf[path] = _parse_json_for_time(json_data)
    else:
        raise ValueError("baseline_run_root does not exist: %s" % baseline_run_root)

    if len(compare_run_roots) != len(compare_run_names):
        raise ValueError(
            "Length mismatch between the number of given paths for comparison and"
            "their run names. %d v/s %d. Each path must have its corresponding run name."
            % (len(compare_run_roots), len(compare_run_names))
        )

    # Read all the comparison related JSON files, one by one, if any.
    compare_perfs = {}
    for compare_run_root, compare_run_name in zip(compare_run_roots, compare_run_names):
        if os.path.isdir(compare_run_root):
            compare_perfs[compare_run_name] = {}

            for path in os.listdir(compare_run_root):
                if os.path.isdir(os.path.join(compare_run_root, path)):
                    compare_perfs[compare_run_name][path] = {}

                    json_path = os.path.join(
                        compare_run_root, path, "benchmark_mean.json"
                    )
                    if os.path.isfile(json_path):
                        with open(json_path, "r") as f:
                            json_data = json.loads(
                                f.read()
                            )  # Storing by the name of the sample

                            compare_perfs[compare_run_name][
                                path
                            ] = _parse_json_for_time(json_data)
        else:
            raise ValueError("compare_run_root does not exist: %s" % compare_run_root)

    results = []

    for sample_name in baseline_perf:
        row_dict = {}

        # Fetch the time and parameters from the JSON for baseline run.
        baseline_run_time = baseline_perf[sample_name]

        row_dict["sample name"] = sample_name
        row_dict["%s time (ms)" % baseline_run_name] = baseline_run_time

        if compare_perfs:
            # Fetch the time from the JSON for all comparison runs.
            for compare_run_name in compare_perfs:
                # Check if the sample was present.
                if sample_name in compare_perfs[compare_run_name]:
                    compare_run_time = compare_perfs[compare_run_name][sample_name]
                else:
                    compare_run_time = None

                row_dict["%s time (ms)" % compare_run_name] = (
                    compare_run_time if compare_run_time else "N/A"
                )

                if baseline_run_time and compare_run_time:
                    speedup = round(compare_run_time / baseline_run_time, 3)
                else:
                    speedup = "N/A"
                row_dict[
                    "%s v/s %s speed-up" % (compare_run_name, baseline_run_name)
                ] = speedup

        results.append(row_dict)

    df = pandas.DataFrame.from_dict(results)

    return df


def main():
    """
    The main function. This will run the comparison function to compare two benchmarking runs.
    """
    parser = argparse.ArgumentParser("Summarize and compare benchmarking runs.")

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="The output directory where you want to store the result summary as a CSV file.",
    )

    parser.add_argument(
        "-b",
        "--baseline-root",
        type=str,
        required=True,
        help="Root folder containing one sub-folder per sample in which benchmark.py styled JSONs"
        " of the baseline runs of those samples are stored.",
    )
    parser.add_argument(
        "-bn",
        "--baseline-name",
        type=str,
        required=True,
        help="The name of the column representing the baseline run in the output table.",
    )
    parser.add_argument(
        "-c",
        "--compare-roots",
        action="append",
        required=False,
        help="Optional. List of folders containing one sub-folder per sample in which benchmark.py"
        " styled JSONs of the comparison runs of those samples are stored.",
    )
    parser.add_argument(
        "-cn",
        "--compare-names",
        action="append",
        required=False,
        help="Optional. List of names of the column representing the comparison runs in the "
        "output table",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        raise ValueError("output-dir does not exist: %s" % args.output_dir)

    if not os.path.isdir(args.baseline_root):
        raise ValueError("baseline-root does not exist: %s" % args.baseline_json)

    if len(args.compare_roots) != len(args.compare_names):
        raise ValueError(
            "Length mismatch between the number of given paths for comparison and"
            "their run names. %d v/s %d. Each path must have its corresponding run name."
            % (len(args.compare_roots), len(args.compare_names))
        )

    logger.info(
        "Summarizing a total of %d runs. All times are in milliseconds"
        % (len(args.compare_roots) + 1)
    )

    df = summarize_runs(
        baseline_run_root=args.baseline_root,
        baseline_run_name=args.baseline_name,
        compare_run_roots=args.compare_roots,
        compare_run_names=args.compare_names,
    )

    csv_path = os.path.join(
        args.output_dir,
        "summarize_runs.%s.csv" % datetime.now(),
    )
    df.to_csv(csv_path)

    logger.info("Wrote comparison CSV to: %s" % csv_path)


assets_dir = "./inference"
parser = get_default_arg_parser(
        "Object Detection sample using CV-CUDA.",
        input_path=os.path.join(assets_dir, "source", "1.jpg"),
        target_img_height=544,
        target_img_width=960,
        supported_backends=["tensorflow", "tensorrt"],
)
parser.add_argument(
    "-c",
    "--confidence_threshold",
    default=0.9,
    type=float,
    help="The confidence threshold for filtering out the detected bounding boxes.",
)

parser.add_argument(
    "-iou",
    "--iou_threshold",
    default=0.2,
    type=float,
    help="The Intersection over Union threshold for NMS.",
)
args = parse_validate_default_args(parser)

cvcuda_perf = CvCudaPerf("object_detection_sample", default_args=args)

def setup_tensort_bindings(trt_model, batch_size, device_id, logger):
    """
    Setups the I/O bindings for a TensorRT engine for the first time.
    :param trt_model: Full path to the generated TensorRT Engine file.
    :param batch_size: The maximum batch size that should be supported in the model.
    :param device_id: The GPU device id on which you want to allocated the buffers.
    :return: A list of output tensors and the index of the first output.
    """
    # For TensorRT, we need to allocate the output data buffers.
    # The input data buffers are already allocated by us.
    output_binding_idx = 0
    output_idx = 0
    output_tensors = []

    # Loop over all the I/O bindings.
    for b_idx in range(trt_model.num_io_tensors):
        # Get various properties associated with the bindings.
        b_name = trt_model.get_tensor_name(b_idx)
        b_shape = list(trt_model.get_tensor_shape(b_name))
        b_dtype = np.dtype(trt.nptype(trt_model.get_tensor_dtype(b_name))).name

        # Append to the appropriate list.
        if trt_model.get_tensor_mode(b_name) == trt.TensorIOMode.OUTPUT:
            # First allocate on device output buffers, using PyTorch.
            # Get the dimensions other the the batch size from the layer shape
            # and set the output layer size for the buffer.
            # The first dimension is the batch size if there is a -1 in its place.
            # If not, then we will simply insert it.
            if b_shape[0] == -1:
                # We need to replace the -1 with the batch size.
                b_shape[0] = batch_size
            elif b_shape[0] == 1 and batch_size == 1:
                # In this case where batch size is 1 and our binding already
                # has it, we will assume it is the batch size dim and not
                # change it.
                pass
            else:
                # Batch size dimension was not present at all. We will need to
                # add it.
                b_shape.insert(0, batch_size)
            
            import cupy as cp
            output = cp.zeros(
                  shape=b_shape,
                  dtype=getattr(cp, b_dtype),
                  # device="cuda:%d" % device_id,
            )
            
            # output = torch.zeros(
            #     size=b_shape,
            #     dtype=getattr(torch, b_dtype),
            #     device="cuda:%d" % device_id,
            # )
            # Since we know the name of our output layer, we will check against
            # it and grab its binding index.
            if b_name == "output":
                output_idx = output_binding_idx

            output_binding_idx += 1
            output_tensors.append(output)

    return output_tensors, output_idx

weights_path = "../weights/yolov7_GB_rtx3060.trt"

class ObjectDetectionTensorRT:
    def __init__(
        self,
        output_dir="./output_dir",
        batch_size=1,
        image_size=(384,640),
        device_id=0,
        cvcuda_perf=cvcuda_perf,
    ):
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.device_id = device_id
        self.cvcuda_perf = cvcuda_perf

        trt_engine_file_path = weights_path

        # Once the TensorRT engine generation is all done, we load it.
        # trt_logger = trt.Logger(trt.Logger.ERROR)
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        # trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        trt.init_libnvinfer_plugins(None, "")
        
        with open(trt_engine_file_path, "rb") as f:
            # Keeping this as a class variable because we want to be able to
            # allocate the output tensors either on its first use or when the
            # batch size changes
            # print(f.read())
            self.trt_model = runtime.deserialize_cuda_engine(f.read())
        # Create execution context.
        self.model = self.trt_model.create_execution_context()
        # We will allocate the output tensors and its bindings either when we
        # use it for the first time or when the batch size changes.
        self.output_tensors, self.output_idx = None, None

        self.logger.info("Using TensorRT as the inference engine.")
        # docs_tag: end_init_objectdetectiontensorrt

    # docs_tag: begin_call_objectdetectiontensorrt
    def __call__(self, tensor):
        self.cvcuda_perf.push_range("inference.tensorrt")

        # Grab the data directly from the pre-allocated tensor.
        input_bindings = [tensor.cuda().__cuda_array_interface__["data"][0]]
        output_bindings = []

        actual_batch_size = tensor.shape[0]

        # Need to allocate the output tensors
        if not self.output_tensors or actual_batch_size != self.batch_size:
            self.output_tensors, self.output_idx = setup_tensort_bindings(
                self.trt_model,
                actual_batch_size,
                self.device_id,
                self.logger,
            )

        for t in self.output_tensors:
            # output_bindings.append(t.data_ptr())
            output_bindings.append(t.data.ptr)
        io_bindings = input_bindings + output_bindings

        # Call inference for implicit batch
        self.model.execute_async(
            actual_batch_size,
            bindings=io_bindings,
            stream_handle=cvcuda.Stream.current.handle,
        )
        
        # print(self.output_tensors[1].shape)

        # boxes = self.output_tensors[0]
        # score = self.output_tensors[1]
        boxes = self.output_tensors[1]
        score = self.output_tensors[2]
        cls_inds = self.output_tensors[3]
        
        # print("boxes.shape - ", boxes.squeeze().shape)
        # print("boxes.shape - ", boxes.shape)
        # print("score.shape - ", score.squeeze()[:, None])
        # print("cls_inds.shape - ", cls_inds.shape)
        self.cvcuda_perf.pop_range()  # inference.tensorrt
        # return torch.cat((boxes.squeeze(), score.swapaxes(0,1), cls_inds.swapaxes(0,1)), -1) 
        return np.column_stack((boxes.squeeze(), score.swapaxes(0,1), cls_inds.swapaxes(0,1))) 