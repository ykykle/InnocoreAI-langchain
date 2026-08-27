import unittest
from unittest.mock import AsyncMock

from agents.controller import AgentController
from core.exceptions import AgentException


def workflow_task(**overrides):
    data = {
        "id": "workflow-test",
        "tenant_id": "tenant-a",
        "user_id": "user-a",
        "input_data": {
            "user_id": "user-a",
            "keywords": ["agents"],
            "validate_citations": True,
            "writing_task": "suggest",
        },
        "agent_results": {},
    }
    data["input_data"].update(overrides)
    return data


class WorkflowFailurePolicyTest(unittest.IsolatedAsyncioTestCase):
    def controller(self):
        controller = AgentController.__new__(AgentController)
        controller._run_agent = AsyncMock()
        return controller

    async def test_hunter_failure_aborts_workflow(self):
        controller = self.controller()
        controller._run_agent.return_value = {
            "status": "failed", "error": "source unavailable"
        }
        with self.assertRaises(AgentException):
            await controller._execute_full_workflow(workflow_task())

    async def test_miner_and_validator_are_best_effort_and_coach_runs(self):
        controller = self.controller()
        miner_calls = 0

        async def run(name, payload, task):
            nonlocal miner_calls
            if name == "hunter":
                return {
                    "status": "success",
                    "papers": [
                        {"title": "A", "abstract": "a"},
                        {"title": "B", "abstract": "b"},
                    ],
                }
            if name == "miner":
                miner_calls += 1
                if miner_calls == 1:
                    return {"success": True, "analysis": "valid"}
                raise AgentException("miner failed")
            if name == "validator":
                raise AgentException("validator failed")
            if name == "coach":
                return {"status": "success", "result": "draft"}
            raise AssertionError(name)

        controller._run_agent.side_effect = run
        result = await controller._execute_full_workflow(workflow_task())
        self.assertEqual(len(result["analysis_reports"]), 1)
        self.assertEqual(result["stages"]["coach"]["status"], "success")
        self.assertTrue(any("Miner" in warning for warning in result["warnings"]))
        self.assertTrue(any("Validator" in warning for warning in result["warnings"]))

    async def test_coach_is_skipped_without_valid_analysis(self):
        controller = self.controller()

        async def run(name, payload, task):
            if name == "hunter":
                return {
                    "status": "success",
                    "papers": [{"title": "A", "abstract": "a"}],
                }
            if name == "miner":
                return {"success": False, "error": "invalid"}
            if name == "validator":
                return {"status": "success", "citations": {}}
            raise AssertionError("Coach must not run")

        controller._run_agent.side_effect = run
        result = await controller._execute_full_workflow(workflow_task())
        self.assertEqual(result["stages"]["coach"]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
