"""
Real-time GPU/CPU Power & Energy-per-Token Monitor.

Samples hardware power draw via NVIDIA NVML (`pynvml`) during inference and integrates
power over time to compute:
  - Average & Peak Power (Watts)
  - Total Energy Consumed (Joules)
  - Energy per Generated Token (Joules / token)
  - Estimated Carbon Footprint (gCO2e per 1,000 tokens, aligned with SDG 9 & SDG 12)

Includes a calibrated TDP + utilization fallback when running on CPUs or GPUs without
NVML power sensor access.
"""

from dataclasses import dataclass
import threading
import time
from typing import List, Optional, Tuple
import psutil
import torch


@dataclass
class EnergyMeasurement:
    """Telemetry captured over a single inference execution window."""

    duration_s: float
    avg_power_watts: float
    peak_power_watts: float
    total_energy_joules: float
    energy_per_token_joules: float
    co2_grams_per_1k_tokens: float
    peak_gpu_memory_mb: float
    sensor_source: str


class EnergyMonitor:
    """Context manager / sampler that measures GPU power and peak VRAM during inference."""

    # Global average grid carbon intensity (~475 gCO2e / kWh = 0.0001319 gCO2e / Joule)
    GRAMS_CO2_PER_JOULE: float = 475.0 / 3_600_000.0

    def __init__(
        self,
        sample_interval_ms: int = 50,
        fallback_gpu_tdp_watts: float = 115.0,
        fallback_cpu_tdp_watts: float = 45.0,
        device_index: int = 0,
    ) -> None:
        self.sample_interval_s = max(0.01, sample_interval_ms / 1000.0)
        self.fallback_gpu_tdp_watts = fallback_gpu_tdp_watts
        self.fallback_cpu_tdp_watts = fallback_cpu_tdp_watts
        self.device_index = device_index

        self._nvml_available = False
        self._nvml_handle = None
        self._pynvml = None
        self._init_nvml()

        self._samples: List[Tuple[float, float]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t_start: float = 0.0
        self._t_end: float = 0.0

    def _init_nvml(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            # Test reading power in milliwatts
            _ = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
            self._pynvml = pynvml
            self._nvml_available = True
        except Exception:
            self._nvml_available = False

    def _read_instant_power_watts(self) -> float:
        if self._nvml_available and self._pynvml is not None:
            try:
                mw = self._pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                return float(mw) / 1000.0
            except Exception:
                pass

        # Fallback estimation based on active device type and CPU/GPU utilization
        if torch.cuda.is_available():
            # Active CUDA execution typically draws ~65-85% of consumer GPU TDP
            return self.fallback_gpu_tdp_watts * 0.75
        cpu_pct = psutil.cpu_percent(interval=None) or 50.0
        return self.fallback_cpu_tdp_watts * (0.3 + 0.7 * (cpu_pct / 100.0))

    def _sampling_loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.perf_counter()
            watts = self._read_instant_power_watts()
            self._samples.append((now, watts))
            self._stop_event.wait(self.sample_interval_s)

    def start(self) -> None:
        """Starts background power sampling and resets peak GPU memory statistics."""
        self._samples.clear()
        self._stop_event.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        else:
            psutil.cpu_percent(interval=None)

        self._t_start = time.perf_counter()
        self._samples.append((self._t_start, self._read_instant_power_watts()))
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

    def stop(
        self,
        num_generated_tokens: int = 1,
        kv_compression_ratio: float = 1.0,
        target_passes: Optional[int] = None,
    ) -> EnergyMeasurement:
        """Stops power sampling and computes integrated energy and memory metrics."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t_end = time.perf_counter()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._samples.append((self._t_end, self._read_instant_power_watts()))

        duration_s = max(1e-6, self._t_end - self._t_start)
        tokens = max(1, num_generated_tokens)

        # Trapezoidal integration of P(t) dt
        total_joules = 0.0
        peak_watts = 0.0
        for i in range(1, len(self._samples)):
            t0, p0 = self._samples[i - 1]
            t1, p1 = self._samples[i]
            dt = max(0.0, t1 - t0)
            total_joules += 0.5 * (p0 + p1) * dt
            peak_watts = max(peak_watts, p0, p1)

        # NOTE: Without an NVML hardware sensor, power is a constant TDP-based estimate,
        # so energy here is simply proportional to wall-clock time. `kv_compression_ratio`
        # and `target_passes` are accepted for API compatibility but intentionally NOT used
        # to scale energy (doing so would bake the expected result into the measurement).
        # Only `sensor_source == "nvml_hardware_sensor"` results are real power measurements.

        avg_watts = total_joules / duration_s
        j_per_tok = total_joules / tokens
        co2_per_1k = j_per_tok * 1000.0 * self.GRAMS_CO2_PER_JOULE

        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            proc = psutil.Process()
            peak_mem_mb = proc.memory_info().rss / (1024 * 1024)

        source = "nvml_hardware_sensor" if self._nvml_available else "tdp_calibrated_model"
        return EnergyMeasurement(
            duration_s=round(duration_s, 4),
            avg_power_watts=round(avg_watts, 2),
            peak_power_watts=round(peak_watts, 2),
            total_energy_joules=round(total_joules, 4),
            energy_per_token_joules=round(j_per_tok, 4),
            co2_grams_per_1k_tokens=round(co2_per_1k, 6),
            peak_gpu_memory_mb=round(peak_mem_mb, 2),
            sensor_source=source,
        )
