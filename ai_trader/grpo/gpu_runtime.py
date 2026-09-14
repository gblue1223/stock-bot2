"""Phase measurements in the training process, including synchronized CUDA peaks."""
import time

import torch


class PhaseMeasurement:
    def __init__(self, device):
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.metrics = {}

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cuda:
            torch.cuda.synchronize(self.device)
        self.metrics['seconds'] = time.perf_counter() - self.started
        if self.cuda:
            free, total = torch.cuda.mem_get_info(self.device)
            self.metrics.update(
                allocated_bytes=torch.cuda.memory_allocated(self.device),
                reserved_bytes=torch.cuda.memory_reserved(self.device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                device_free_bytes=free, device_total_bytes=total)
            self.metrics['peak_allocated_gib'] = self.metrics['peak_allocated_bytes'] / 1024**3
            self.metrics['peak_reserved_gib'] = self.metrics['peak_reserved_bytes'] / 1024**3

    def throughput(self, samples):
        self.metrics['samples'] = int(samples)
        self.metrics['samples_per_second'] = samples / max(self.metrics['seconds'], 1e-9)
        return self.metrics
