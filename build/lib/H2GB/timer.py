import time
from contextlib import ContextDecorator
from typing import NamedTuple
import torch
import statistics

import importlib.util

# Convenience utility for recording runtime statistics during execution
#   and reporting the results after the run.
runtime_statistics = None
runtime_statistics_enabled = False

# prevents runtime statistics from being enabled.
performance_stats_enabled = True


def is_performance_stats_enabled():
    global performance_stats_enabled
    return performance_stats_enabled


class RuntimeStatisticsCUDA:
    # _instance = None

    # def __new__(cls, name):
    #     if cls._instance is None:
    #         cls._instance = super(RuntimeStatisticsCUDA, cls).__new__(cls)
    #         # Initialize your object here instead of __init__
    #         cls._instance.name = name
    #     return cls._instance

    def __init__(self, name: str):
        self.stat_lists = dict()
        self.stat_lists_cuda = dict()
        self.name = name
        self.epoch_counter = 0
        self.cuda_timer_lists = dict()
        self.last_event = None
        self.cuda_times = dict()
        self.cuda_timer_start = dict()
        self.cuda_timer_end = dict()

    def start_epoch(self):
        self.epoch_counter += 1

    def get_last_event(self):
        return self.last_event

    def start_region(self, region_name, use_event=None):
        if not runtime_statistics_enabled:
            return
        if use_event is not None:
            self.cuda_timer_start[region_name] = use_event
            self.last_event = use_event
        else:
            self.cuda_timer_start[region_name] = torch.cuda.Event(
                enable_timing=True)
            self.last_event = self.cuda_timer_start[region_name]
            self.cuda_timer_start[region_name].record()

    def end_region(self, region_name, use_event=None):
        if not runtime_statistics_enabled:
            return
        if use_event is not None:
            self.cuda_timer_end[region_name] = use_event
            self.last_event = use_event
        else:
            self.cuda_timer_end[region_name] = torch.cuda.Event(
                enable_timing=True)
            self.last_event = self.cuda_timer_end[region_name]
            self.cuda_timer_end[region_name].record()
        if region_name not in self.cuda_timer_lists:
            self.cuda_timer_lists[region_name] = []
        self.cuda_timer_lists[region_name].append(
            (self.cuda_timer_start[region_name], self.cuda_timer_end[region_name]))

    def end_epoch(self):
        torch.cuda.synchronize()
        for x in self.cuda_timer_lists.keys():
            total = self.cuda_timer_lists[x][0][0].elapsed_time(
                self.cuda_timer_lists[x][0][1])
            for y in self.cuda_timer_lists[x][1:]:
                total += y[0].elapsed_time(y[1])
            if x not in self.cuda_times:
                self.cuda_times[x] = []
            if self.epoch_counter > 1:
                self.cuda_times[x].append(total)
        self.cuda_timer_lists = dict()
        self.cuda_timer_start = dict()
        self.cuda_timer_end = dict()

    def report_stats(self, display_keys=None):
        rows = []
        for x in sorted(self.cuda_times.keys()):
            print_name = x
            if display_keys is not None and x not in display_keys:
                continue
            elif display_keys is not None:
                print_name = display_keys[x]
            row = []
            if len(self.cuda_times[x]) < 2:
                row = [print_name, "N/A", "N/A"]
                if len(self.cuda_times[x]) == 1:
                    row = [print_name, statistics.mean(
                        self.cuda_times[x]), "N/A"]
            else:
                row = [print_name, statistics.mean(
                    self.cuda_times[x]), statistics.stdev(self.cuda_times[x])]
            rows.append(row)
        exists = importlib.util.find_spec("prettytable") is not None
        if not exists:
            print("activity, mean time (ms), stdev")
            for row in rows:
                print(", ".join([str(a) for a in row]))
        else:
            import prettytable
            tab = prettytable.PrettyTable()
            num_samples = -1
            for x in sorted(self.cuda_times.keys()):
                assert num_samples < 0 or num_samples == len(
                    self.cuda_times[x])
                num_samples = len(self.cuda_times[x])
            tab.field_names = [
                "Activity ("+self.name+")", "Mean time (ms) (over " + str(num_samples) + " epochs)", "Stdev"]
            for x in rows:
                tab.add_row(x)
            # for x in sorted(self.cuda_times.keys()):
            #    print_name = x
            #    if display_keys is not None and x not in display_keys:
            #        continue
            #    elif display_keys is not None:
            #        print_name = display_keys[x]
            #    if len(self.cuda_times[x]) < 2:
            #        tab.add_row([print_name, "N/A", "N/A"])
            #    else:
            #        tab.add_row([print_name, statistics.mean(self.cuda_times[x]), statistics.stdev(self.cuda_times[x])])
            print(tab.get_string(sortby=tab.field_names[1]))

        #print("===Showing runtime stats for: " + self.name + " ===")

        # for x in sorted(self.cuda_times.keys()):
        #    if len(self.cuda_times[x]) < 2:
        #        print (x + ": N/A")
        #    else:
        #        print (x + " Mean: " + str(statistics.mean(self.cuda_times[x])) + " Stdev: " + str(statistics.stdev(self.cuda_times[x])))
        return str(rows)

    def clear_stats(self):
        self.cuda_times = dict()
        self.cuda_timer_lists = dict()
        self.cuda_timer_start = dict()
        self.cuda_timer_end = dict()


runtime_stats_cuda = RuntimeStatisticsCUDA("H2GB") 


def setup_runtime_stats(args):
    global runtime_statistics
    global performance_stats_enabled
    runtime_statistics = RuntimeStatistics("")
    if args.train_sampler == 'NeighborSampler':
        sampler = "PyG"
    else:
        sampler = "SALIENT"
    model = args.model_name
    dataset = args.dataset_name
    if args.performance_stats:
        performance_stats_enabled = True
    else:
        performance_stats_enabled = False
    runtime_stats_cuda.name = " ".join([sampler, dataset, model])


def enable_runtime_stats():
    if not performance_stats_enabled:
        return
    global runtime_statistics_enabled
    runtime_statistics_enabled = True


def disable_runtime_stats():
    if not performance_stats_enabled:
        return
    global runtime_statistics_enabled
    runtime_statistics_enabled = False


def start_runtime_stats_epoch():
    runtime_statistics.start_epoch()


def report_runtime_stats(logger=None):
    # if runtime_statistics is not None:
    #    runtime_statistics.report_stats()
    if is_performance_stats_enabled() and runtime_stats_cuda is not None:
        string_output = runtime_stats_cuda.report_stats(
            {'total': 'Total', 'data_transfer': 'Data Transfer', 'sampling': 'Sampling + Slicing', 'train': 'Train'})
        if logger is not None:
            logger(('performance_breakdown_stats', string_output))


def append_runtime_stats(name, value):
    if runtime_statistics is not None and runtime_statistics_enabled:
        runtime_statistics.append_stat(name, value)


class RuntimeStatistics:
    def __init__(self, name: str):
        self.stat_lists = dict()
        self.name = name
        self.epoch_counter = 0

    def start_epoch(self):
        self.epoch_counter += 1

    def append_stat(self, name, value):
        # skip the first epoch.
        if self.epoch_counter == 1:
            return
        if name not in self.stat_lists:
            self.stat_lists[name] = []
        self.stat_lists[name].append(value)

    def report_stats(self):
        print("===Showing runtime stats for: " + self.name + " ===")
        for x in sorted(self.stat_lists.keys()):
            if len(self.stat_lists[x]) == 0:
                print(x + ": N/A")
            else:
                print(x + " Mean: " + str(statistics.mean(self.stat_lists[x])) + " Stdev: " + str(
                    statistics.stdev(self.stat_lists[x])))

    def clear_stats(self):
        self.stat_lists = dict()


class TimerResult(NamedTuple):
    name: str
    nanos: int

    def __str__(self):
        return f'{self.name} took {self.nanos / 1e9} sec'


class CUDAAggregateTimer:
    def __init__(self, name: str):
        self.name = name
        self.timer_list = []
        self._start = None
        self._end = None

    def get_start(self):
        return self._start

    def get_end(self):
        return self._end

    def start(self, timer=None):
        if timer is None:
            self._start = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._start = timer

    def end(self, timer=None):
        # print(torch.cuda.current_stream())
        # print(stream)
        if timer is None:
            self._end = torch.cuda.Event(enable_timing=True)
            self._end.record()
        else:
            self._end = timer
            # self._end.record(stream)
        self.timer_list.append((self._start, self._end))

    def report(self, do_print=False):
        torch.cuda.synchronize()
        total_time = self.timer_list[0][0].elapsed_time(self.timer_list[0][1])
        for x in self.timer_list[1:]:
            total_time += x[0].elapsed_time(x[1])
        if do_print:
            print("CUDA Aggregate (" + self.name + "): "+str(total_time)+" msec")
        return total_time


class Timer(ContextDecorator):
    def __init__(self, name: str, fn=print):
        self.name = name
        self._fn = fn

    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        self.stop_ns = None
        return self

    def stop(self):
        self.stop_ns = time.perf_counter_ns()

    def __exit__(self, *_):
        if self.stop_ns is None:
            self.stop()
        nanos = self.stop_ns - self.start_ns
        self._fn(TimerResult(self.name, nanos))
