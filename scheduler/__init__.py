"""GPU Scheduler for FamilyGPU Orchestrator.

The scheduler is the core decision-making component. It accepts job
requests, selects the best available account, creates leases, and
manages the job lifecycle including failover.

Components:
  - request.py: Job request model and validation
  - selector.py: Account selection algorithm with scoring
  - leases.py: Lease lifecycle management
  - quota.py: Quota enforcement and checking
  - failover.py: Failover logic when jobs fail
  - scoring.py: Account scoring algorithm
  - autoloop.py: Background daemon for continuous auto-scheduling
"""

from scheduler.request import JobRequest, JobRequestResult, FailureReason
from scheduler.selector import AccountSelector
from scheduler.leases import LeaseManager
from scheduler.quota import QuotaEnforcer
from scheduler.failover import FailoverManager
from scheduler.autoloop import AutoLoop, AutoLoopConfig, AutoLoopStats

__all__ = [
    "JobRequest",
    "JobRequestResult",
    "FailureReason",
    "AccountSelector",
    "LeaseManager",
    "QuotaEnforcer",
    "FailoverManager",
    "AutoLoop",
    "AutoLoopConfig",
    "AutoLoopStats",
]
