"""
Audio Package — Speech-to-Text, Text-to-Speech, Resource Scheduling, and Audio DSP Utilities.
"""

from audio.utils import (
    get_effective_input_device,
    get_supported_sample_rate,
    get_optimal_channels,
    resample_to_16k,
    list_audio_devices,
    test_mic_hardware,
)
from audio.scheduler import (
    AudioStage,
    ResourceScheduler,
    scheduler,
    set_thread_high_priority,
    pin_current_thread_to_cores,
)

__all__ = [
    # Audio Utils
    "get_effective_input_device",
    "get_supported_sample_rate",
    "get_optimal_channels",
    "resample_to_16k",
    "list_audio_devices",
    "test_mic_hardware",
    # Scheduler
    "AudioStage",
    "ResourceScheduler",
    "scheduler",
    "set_thread_high_priority",
    "pin_current_thread_to_cores",
]
