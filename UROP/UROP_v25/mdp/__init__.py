from __future__ import annotations

# Re-export Isaac Lab built-in MDP terms/configs first. Custom UROP-v25 terms below
# intentionally override observation/reward names where needed.
try:
    from isaaclab.envs.mdp import *  # noqa: F401,F403
except Exception:
    pass

try:
    from isaaclab.envs.mdp.actions import JointPositionActionCfg  # type: ignore # noqa: F401
except Exception:
    try:
        from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg  # type: ignore # noqa: F401
    except Exception:
        pass

try:
    from isaaclab.envs.mdp.commands import NullCommandCfg  # type: ignore # noqa: F401
except Exception:
    try:
        from isaaclab.envs.mdp.commands.commands_cfg import NullCommandCfg  # type: ignore # noqa: F401
    except Exception:
        pass

from .observations import *  # noqa: F401,F403
from .rewards import *  # noqa: F401,F403
from .terminations import *  # noqa: F401,F403
from .events import *  # noqa: F401,F403
from .curriculum import *  # noqa: F401,F403
