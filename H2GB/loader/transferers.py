# from typing import List
# #from collections.abc import Iterator
# from typing import Iterator
# import torch

# from .samplers import ProtoBatch, PreparedBatch
# from .utils import append_runtime_stats, Timer, runtime_stats_cuda
# import time


# class DeviceIterator(Iterator[List[PreparedBatch]]):
#     '''
#     Abstract class that returns PreparedBatch on devices (GPUs)
#     '''
#     devices: List[torch.cuda.device]

#     def __init__(self, devices):
#         assert len(devices) > 0
#         self.devices = devices


# class DevicePrefetcher(DeviceIterator):
#     def __init__(self, devices, it: Iterator[PreparedBatch]):
#         super().__init__(devices)

#         self.it = it
#         self.streams = [torch.cuda.Stream(device) for device in devices]
#         self.next = []
#         self.sampling_times = []
#         self.record_stream_times = []
#         self.start_prefetch_times = []
#         self.wait_stream_times = []
#         self.preload(False)

#     def preload(self, timing=True):
#         self.next = []
#         for device, stream in zip(self.devices, self.streams):
#             timer_start = time.perf_counter_ns()
#             batch = next(self.it, None)
#             timer_end = time.perf_counter_ns()
#             if batch is None:
#                 append_runtime_stats("total:load_batch:sampling", sum(
#                     self.sampling_times)/1000000)
#                 self.sampling_times = []
#                 append_runtime_stats("total:load_batch:data_transfer:start_nonblocking_prefetch", sum(
#                     self.start_prefetch_times)/1000000)
#                 self.start_prefetch_times = []
#                 break

#             timer_start = time.perf_counter_ns()
#             with torch.cuda.stream(stream):
#                 self.next.append(batch.to(device, non_blocking=True))
#             timer_end = time.perf_counter_ns()
#             self.start_prefetch_times.append(timer_end-timer_start)

#     def __next__(self):
#         runtime_stats_cuda.start_region(
#             "data_transfer", runtime_stats_cuda.get_last_event())

#         timer_start = time.perf_counter_ns()
#         cur_streams = [torch.cuda.current_stream(
#             device) for device in self.devices]

#         for cur_stream, stream in zip(cur_streams, self.streams):
#             cur_stream.wait_stream(stream)
#         runtime_stats_cuda.end_region("data_transfer")

#         runtime_stats_cuda.start_region(
#             "sampling", runtime_stats_cuda.get_last_event())

#         ret = self.next
#         timer_end = time.perf_counter_ns()
#         self.wait_stream_times.append(timer_end-timer_start)
#         if not ret:
#             torch.cuda.synchronize()
#             append_runtime_stats("total:load_batch:data_transfer:wait_stream", sum(
#                 self.wait_stream_times)/1000000)
#             self.wait_stream_times = []
#             append_runtime_stats("total:load_batch:data_transfer:record_stream", sum(
#                 self.record_stream_times)/1000000)
#             self.record_stream_times = []
#             raise StopIteration

#         # TODO: this might be a bit incorrect
#         #
#         # in theory, we want to record this event after all the
#         # training computation on the default stream

#         timer_start = time.perf_counter_ns()
#         for cur_stream, batch in zip(cur_streams, ret):
#             batch.record_stream(cur_stream)
#         timer_stop = time.perf_counter_ns()
#         self.record_stream_times.append(timer_stop-timer_start)

#         self.preload()
#         return ret


# class DeviceTransferer(DeviceIterator):
#     def __init__(self, devices, it: Iterator[PreparedBatch]):
#         super().__init__(devices)

#         self.it = it

#     def __next__(self):
#         ret = [batch.to(device, non_blocking=True)
#                for device, batch in zip(self.devices, self.it)]
#         if len(ret) == 0:
#             raise StopIteration

#         return ret


# class DeviceSlicerTransferer(DeviceIterator):
#     # NOTE: This class only exists to provide functionality
#     #       that we used to have and no longer need (DATA_ON_MAIN).
#     #       You likely do not need to use this.
#     # NOTE: x and y can be GPU tensors too!
#     def __init__(self, devices, x: torch.Tensor, y: torch.Tensor,
#                  it: Iterator[ProtoBatch]):
#         super().__init__(devices)

#         self.x = x
#         self.y = y
#         self.it = it

#     def __next__(self):
#         ret = [PreparedBatch.from_proto_batch(
#             self.x, self.y, proto_batch).to(device, non_blocking=True)
#             for device, proto_batch in zip(self.devices, self.it)]

#         if len(ret) == 0:
#             raise StopIteration

#         return ret
