import logging
import threading

import torch


class GPUManager:
    """
    Thread-safe manager that tracks reserved VRAM across GPU devices.

    Call ``acquire(estimated_bytes)`` before moving a model to GPU – it returns
    the device string (e.g. ``"cuda:0"``) of a GPU with enough headroom and
    marks the VRAM as reserved.  Call ``release(device, estimated_bytes)`` when
    the model is moved back to CPU.

    If no device has enough room the calling thread blocks until another thread
    releases VRAM.
    """

    def __init__(self, vram_safety_fraction: float = 0.80):
        """
        Args:
            vram_safety_fraction: Fraction of each GPU's *total* VRAM that we
                allow ourselves to reserve (0-1).  Keeps a safety buffer so
                PyTorch internals / fragmentation don't cause OOM.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("GPUManager requires CUDA but no GPU is available")

        self._num_devices = torch.cuda.device_count()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        # Per-device bookkeeping
        self._total_bytes: list[int] = []
        self._usable_bytes: list[int] = []
        self._reserved_bytes: list[int] = []

        for i in range(self._num_devices):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory
            usable = int(total * vram_safety_fraction)
            self._total_bytes.append(total)
            self._usable_bytes.append(usable)
            self._reserved_bytes.append(0)
            logging.info(
                "GPUManager: cuda:%d — %.1f GB total, %.1f GB usable (%.0f%% margin)",
                i,
                total / 1e9,
                usable / 1e9,
                (1 - vram_safety_fraction) * 100,
            )

    def acquire(self, estimated_bytes: int) -> str:
        """Reserve *estimated_bytes* on the least-loaded GPU that can fit it.

        Blocks until a device has room.  Returns a device string like
        ``"cuda:0"``.
        """
        with self._condition:
            while True:
                best_idx = self._find_best_device(estimated_bytes)
                if best_idx is not None:
                    self._reserved_bytes[best_idx] += estimated_bytes
                    device = f"cuda:{best_idx}"
                    logging.debug(
                        "GPUManager: acquired %.1f MB on %s (%.1f / %.1f MB reserved)",
                        estimated_bytes / 1e6,
                        device,
                        self._reserved_bytes[best_idx] / 1e6,
                        self._usable_bytes[best_idx] / 1e6,
                    )
                    return device
                # No device can fit this right now – wait for a release.
                logging.debug(
                    "GPUManager: waiting for %.1f MB to become available",
                    estimated_bytes / 1e6,
                )
                self._condition.wait()

    def release(self, device: str, estimated_bytes: int) -> None:
        """Release a previous reservation."""
        idx = self._device_index(device)
        with self._condition:
            self._reserved_bytes[idx] = max(0, self._reserved_bytes[idx] - estimated_bytes)
            logging.debug(
                "GPUManager: released %.1f MB on %s (%.1f / %.1f MB reserved)",
                estimated_bytes / 1e6,
                device,
                self._reserved_bytes[idx] / 1e6,
                self._usable_bytes[idx] / 1e6,
            )
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_best_device(self, needed: int) -> int | None:
        """Return index of the GPU with the most free (unreserved) usable
        VRAM that can still fit *needed* bytes, or ``None``."""
        best_idx = None
        best_free = -1
        for i in range(self._num_devices):
            free = self._usable_bytes[i] - self._reserved_bytes[i]
            if free >= needed and free > best_free:
                best_free = free
                best_idx = i
        return best_idx

    @staticmethod
    def _device_index(device: str) -> int:
        if device == "cuda":
            return 0
        # "cuda:N"
        return int(device.split(":")[1])
