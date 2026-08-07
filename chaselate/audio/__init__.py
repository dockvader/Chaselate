"""Audio capture for Windows: WASAPI loopback (system output) or a microphone."""

from .devices import DeviceInfo, available_backends, default_device, list_devices, resolve_device
from .resample import StreamResampler, downmix_mono
from .capture import AudioCapture, CaptureError

__all__ = [
    "AudioCapture",
    "CaptureError",
    "DeviceInfo",
    "StreamResampler",
    "available_backends",
    "default_device",
    "downmix_mono",
    "list_devices",
    "resolve_device",
]
