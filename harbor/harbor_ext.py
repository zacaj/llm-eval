"""Harbor agent wrappers, loaded by harbor.sh as custom agents (-a harbor_ext:<Class>,
with this directory on PYTHONPATH).
"""
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.terminus_2.terminus_2 import Terminus2

# Body of the prompt-logger proxy's eval-mode rejection (see its "Eval mode" docstring)
BUDGET_MSG = "eval token budget exhausted"


class BudgetedTerminus2(Terminus2):
    """terminus-2 that treats the proxy's budget rejection as the end of the episode.

    Harbor only verifies a trial after an agent timeout or non-zero agent exit;
    terminus-2 runs in-process, so the rejection surfaces as a raw litellm
    BadRequestError and the trial would go ungraded. Re-raising it as
    NonZeroAgentExitCodeError gets the workspace graded, same as Claude Code
    (an installed agent) gets when its CLI exits on the same error. name() is
    inherited, so results still say "terminus-2".
    """

    async def run(self, instruction, environment, context) -> None:
        try:
            await super().run(instruction, environment, context)
        except Exception as exc:
            if BUDGET_MSG in str(exc):
                raise NonZeroAgentExitCodeError(BUDGET_MSG) from exc
            # test-calls tasks: after the last allowed `run-tests` call the
            # container kills every agent-user process, including terminus-2's
            # tmux session, so whatever error that causes here means "done"
            try:
                marker = await environment.exec(
                    "test -f /var/lib/run-tests/limit_reached", user="root", timeout_sec=30
                )
            except Exception:
                marker = None
            if marker is not None and marker.return_code == 0:
                raise NonZeroAgentExitCodeError("run-tests call limit reached") from exc
            raise
