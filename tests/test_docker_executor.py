"""Unit and integration tests for Phase 2 Docker container sandboxing."""

import os
import subprocess
from unittest.mock import MagicMock, patch
import pytest

from app.tools.code_executor import (
    DockerExecutor,
    SubprocessExecutor,
    DemoExecutor,
    get_executor,
    _docker_mount_path,
)
from app.state import ExecutionResult


class TestDockerExecutor:
    """Test DockerExecutor construction, security flags, command building, and execution."""

    def test_docker_executor_properties(self):
        executor = DockerExecutor(
            image="test-image:latest",
            cpus=1.5,
            memory="256m",
            pids_limit=32,
            user="2000:2000",
        )
        assert executor.backend_name == "Docker Container Sandbox (Phase 2 — Isolated)"
        assert executor.is_sandboxed is True
        assert executor.image == "test-image:latest"
        assert executor.cpus == 1.5
        assert executor.memory == "256m"
        assert executor.pids_limit == 32
        assert executor.user == "2000:2000"

    def test_build_docker_command_security_flags(self):
        executor = DockerExecutor(
            image="data-analyst-sandbox:latest",
            cpus=2.0,
            memory="512m",
            pids_limit=64,
            user="1000:1000",
        )
        cmd = executor.build_docker_command(
            container_name="test_sandbox_123",
            script_host_dir="/tmp/script",
            script_filename="analysis.py",
            dataset_host_dir="/tmp/data",
            dataset_filename="data.csv",
            output_host_dir="/tmp/output",
        )

        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert "--name" in cmd
        assert cmd[cmd.index("--name") + 1] == "test_sandbox_123"
        assert "--rm" in cmd
        assert "--network" in cmd
        assert cmd[cmd.index("--network") + 1] == "none"  # Disabled network access
        assert "--user" in cmd
        assert cmd[cmd.index("--user") + 1] == "1000:1000"  # Non-root user
        assert "--cpus" in cmd
        assert cmd[cmd.index("--cpus") + 1] == "2.0"  # CPU limit
        assert "--memory" in cmd
        assert cmd[cmd.index("--memory") + 1] == "512m"  # Memory limit
        assert "--pids-limit" in cmd
        assert cmd[cmd.index("--pids-limit") + 1] == "64"  # Process count limit
        assert "--security-opt" in cmd
        assert "no-new-privileges:true" in cmd
        assert "--cap-drop" in cmd
        assert "ALL" in cmd  # Dropped capabilities
        assert "-w" in cmd
        assert "/workspace/output" in cmd

        # Verify mount permissions: data is ro, script is ro, output is rw
        mounts = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-v"]
        assert any("/workspace/data:ro" in m for m in mounts)
        assert any("/workspace/script:ro" in m for m in mounts)
        assert any("/workspace/output:rw" in m for m in mounts)

        # Verify entrypoint
        assert cmd[-2] == "python"
        assert cmd[-1] == "/workspace/script/analysis.py"

    def test_docker_mount_path_normalization(self):
        norm = _docker_mount_path("C:\\Users\\test\\data")
        assert "\\" not in norm
        assert "C:/Users/test/data" in norm

    def test_docker_unavailable_fails_explicitly_without_host_execution(self, tmp_path):
        executor = DockerExecutor()
        dataset_file = str(tmp_path / "test.csv")
        with open(dataset_file, "w") as f:
            f.write("a,b\n1,2\n")

        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir, exist_ok=True)

        with patch.object(DockerExecutor, "is_docker_available", return_value=False):
            res = executor.execute("print('hello')", dataset_file, out_dir)

            assert res.success is False
            assert res.exit_code == -1
            assert "Docker daemon is unavailable" in res.error_summary
            assert "NOT run on host" in res.error_summary

    def test_mocked_docker_execution_success(self, tmp_path):
        executor = DockerExecutor()
        dataset_file = str(tmp_path / "test.csv")
        with open(dataset_file, "w") as f:
            f.write("a,b\n1,2\n")

        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir, exist_ok=True)

        # Create dummy output file in out_dir to simulate container output
        test_chart = os.path.join(out_dir, "chart.png")
        with open(test_chart, "w") as f:
            f.write("dummy-png")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"results": {"metric": 1.0}}'
        mock_proc.stderr = ""

        with patch.object(DockerExecutor, "is_docker_available", return_value=True), \
             patch("subprocess.run", return_value=mock_proc):
            res = executor.execute("print('sandbox')", dataset_file, out_dir, timeout=30)

            assert res.success is True
            assert res.exit_code == 0
            assert "chart.png" in res.generated_files

    def test_mocked_docker_execution_timeout(self, tmp_path):
        executor = DockerExecutor()
        dataset_file = str(tmp_path / "test.csv")
        with open(dataset_file, "w") as f:
            f.write("a,b\n1,2\n")

        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir, exist_ok=True)

        with patch.object(DockerExecutor, "is_docker_available", return_value=True), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="docker run", timeout=10)):
            res = executor.execute("import time; time.sleep(100)", dataset_file, out_dir, timeout=10)

            assert res.success is False
            assert "timed out after 10 seconds" in res.error_summary
            assert res.exit_code == -1


class TestExecutorResolution:
    """Test get_executor backend factory behavior and transparency."""

    def test_get_executor_explicit_docker(self):
        executor = get_executor(backend="docker")
        assert isinstance(executor, DockerExecutor)
        assert executor.is_sandboxed is True
        assert "Docker Container Sandbox" in executor.backend_name

    def test_get_executor_explicit_subprocess(self):
        executor = get_executor(backend="subprocess")
        assert isinstance(executor, SubprocessExecutor)
        assert executor.is_sandboxed is False
        assert "not sandboxed" in executor.backend_name

    def test_get_executor_auto_when_docker_down(self, tmp_path):
        with patch.object(DockerExecutor, "is_docker_available", return_value=False):
            executor = get_executor(backend="auto")
            assert isinstance(executor, DockerExecutor)
            dataset_file = tmp_path / "data.csv"
            dataset_file.write_text("x\n1\n", encoding="utf-8")
            result = executor.execute("print('must not run on host')", str(dataset_file), str(tmp_path), timeout=1)
            assert result.success is False
            assert "NOT run on host" in result.error_summary

    def test_get_executor_auto_when_docker_up(self):
        with patch.object(DockerExecutor, "is_docker_available", return_value=True):
            executor = get_executor(backend="auto")
            assert isinstance(executor, DockerExecutor)
            assert executor.is_sandboxed is True

    def test_demo_executor_reports_not_sandboxed_on_host(self):
        demo = DemoExecutor(use_docker=False)
        assert demo.is_sandboxed is False
        assert "Local Subprocess" in demo.backend_name

