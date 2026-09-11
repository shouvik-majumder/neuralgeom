"""neuralgeom.tasks.timing -- the cue-triggered lick-timing task."""
from .config import (ObservationConfig, SchedulerConfig, TimingTaskConfig,
                     VARIANTS, make_config)
from .generator import Phase, StepResult, TrialGenerator
from .scheduler import DelayScheduler
from .monitor import (JSONLMonitor, MemoryMonitor, Monitor, MonitorList,
                      read_jsonl, records_to_arrays)

__all__ = ["TimingTaskConfig", "SchedulerConfig", "ObservationConfig",
           "VARIANTS", "make_config", "TrialGenerator", "StepResult", "Phase",
           "DelayScheduler", "Monitor", "MonitorList", "MemoryMonitor",
           "JSONLMonitor", "read_jsonl", "records_to_arrays"]


def __getattr__(name):
    # gymnasium is an optional extra; importing the env should only fail for
    # someone who actually asks for it.
    if name == "TimingTaskEnv":
        from .env import TimingTaskEnv
        return TimingTaskEnv
    raise AttributeError(name)
