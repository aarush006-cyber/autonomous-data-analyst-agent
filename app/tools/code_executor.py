"""Code execution service abstraction.

Phase 1: subprocess-based execution with security restrictions.
Phase 2: Replace SubprocessExecutor with DockerExecutor.

WARNING — DEVELOPMENT ONLY:
The Phase 1 SubprocessExecutor runs generated Python code in a local subprocess.
It applies basic restrictions (timeouts, path validation, import filtering) but does
NOT provide a true sandbox. Do not use this in production with untrusted inputs.
"""

from __future__ import annotations

import abc
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from app.state import ExecutionResult

logger = logging.getLogger(__name__)

# Imports that must NOT appear in generated code
BLOCKED_IMPORTS = [
    "subprocess", "os.system", "shutil.rmtree", "socket", "http",
    "urllib", "requests", "paramiko", "ftplib", "smtplib",
    "ctypes", "importlib", "eval", "exec", "__import__",
]

MAX_STDOUT = 200_000   # bytes
MAX_STDERR = 50_000    # bytes


class ExecutorInterface(abc.ABC):
    """Abstract execution backend. Implement for subprocess (Phase 1) or Docker (Phase 2)."""

    @abc.abstractmethod
    def execute(
        self,
        code: str,
        dataset_path: str,
        output_dir: str,
        timeout: int = 120,
    ) -> ExecutionResult:
        """Execute analysis code and return the result."""
        ...

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Human-readable name of the execution backend."""
        ...

    @property
    def is_sandboxed(self) -> bool:
        """Whether the backend provides true isolation."""
        return False


def validate_code_safety(code: str, max_size_kb: int = 100) -> list[str]:
    """Check generated code for obvious safety violations. Returns list of warnings."""
    warnings: list[str] = []

    size_kb = len(code.encode("utf-8")) / 1024
    if size_kb > max_size_kb:
        warnings.append(f"Code exceeds size limit ({size_kb:.1f} KB > {max_size_kb} KB)")

    for blocked in BLOCKED_IMPORTS:
        pattern = r"\b" + re.escape(blocked).replace(r"\.", r"\s*\.\s*") + r"\b"
        if re.search(pattern, code):
            warnings.append(f"Blocked import/function detected: {blocked}")

    # Check for file access outside expected paths
    if "open(" in code and ("/etc" in code or "C:\\" in code.replace("C:\\Users", "")):
        warnings.append("Suspicious file access pattern detected")

    return warnings


def _sanitize_path(path: str, allowed_base: str) -> str:
    """Prevent path traversal attacks."""
    resolved = os.path.realpath(path)
    base = os.path.realpath(allowed_base)
    if not resolved.startswith(base):
        raise ValueError(f"Path traversal detected: {path}")
    return resolved


class SubprocessExecutor(ExecutorInterface):
    """Phase 1 executor: runs code in a local subprocess.
    
    ⚠️  DEVELOPMENT ONLY — NOT A SECURE SANDBOX.
    This executor applies timeouts and basic import filtering but runs
    code directly on the host. Phase 2 must replace this with Docker.
    """

    @property
    def backend_name(self) -> str:
        return "Local Subprocess (DEVELOPMENT ONLY — not sandboxed)"

    @property
    def is_sandboxed(self) -> bool:
        return False

    def execute(
        self,
        code: str,
        dataset_path: str,
        output_dir: str,
        timeout: int = 120,
    ) -> ExecutionResult:
        """Execute Python code in a subprocess with timeout."""
        start = time.time()

        # Validate paths
        dataset_path = os.path.realpath(dataset_path)
        output_dir = os.path.realpath(output_dir)

        if not os.path.isfile(dataset_path):
            return ExecutionResult(
                success=False,
                error_summary=f"Dataset not found: {dataset_path}",
                exit_code=-1,
                duration_seconds=0,
            )

        os.makedirs(output_dir, exist_ok=True)

        # Safety check
        safety_warnings = validate_code_safety(code)
        if any("Blocked" in w for w in safety_warnings):
            return ExecutionResult(
                success=False,
                error_summary=f"Code rejected: {'; '.join(safety_warnings)}",
                exit_code=-1,
                duration_seconds=0,
            )

        # Prepare the execution script
        # Inject DATASET_PATH and OUTPUT_DIR as safely quoted constants
        header = (
            f"DATASET_PATH = {repr(dataset_path)}\n"
            f"OUTPUT_DIR = {repr(output_dir)}\n\n"
        )
        full_code = header + code

        # Write to temp file
        tmp_dir = tempfile.mkdtemp(prefix="analyst_exec_")
        script_path = os.path.join(tmp_dir, "analysis_script.py")
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(full_code)

            # Execute — do NOT pass any environment variables with secrets
            clean_env = {
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "SYSTEMDRIVE": os.environ.get("SYSTEMDRIVE", "C:"),
                "PROGRAMDATA": os.environ.get("PROGRAMDATA", "C:\\ProgramData"),
                "ALLUSERSPROFILE": os.environ.get("ALLUSERSPROFILE", "C:\\ProgramData"),
                "WINDIR": os.environ.get("WINDIR", os.environ.get("SYSTEMROOT", "C:\\Windows")),
                "TEMP": os.environ.get("TEMP", tmp_dir),
                "TMP": os.environ.get("TMP", tmp_dir),
            }
            # Add USERPROFILE for Windows Python to find its packages
            if "USERPROFILE" in os.environ:
                clean_env["USERPROFILE"] = os.environ["USERPROFILE"]
            if "APPDATA" in os.environ:
                clean_env["APPDATA"] = os.environ["APPDATA"]
            if "LOCALAPPDATA" in os.environ:
                clean_env["LOCALAPPDATA"] = os.environ["LOCALAPPDATA"]

            proc = subprocess.run(
                [sys.executable or "python", script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=output_dir,
                env=clean_env,
            )

            # Defensive cleanup: remove accidental Windows cache path if created in cwd
            stray_sysdrive = os.path.join(output_dir, "%SystemDrive%")
            if os.path.isdir(stray_sysdrive):
                shutil.rmtree(stray_sysdrive, ignore_errors=True)

            duration = time.time() - start

            stdout = proc.stdout[:MAX_STDOUT] if proc.stdout else ""
            stderr = proc.stderr[:MAX_STDERR] if proc.stderr else ""

            # Collect generated files
            generated_files = []
            for f in os.listdir(output_dir):
                fpath = os.path.join(output_dir, f)
                if os.path.isfile(fpath):
                    # Only accept expected file types
                    ext = os.path.splitext(f)[1].lower()
                    if ext in (".png", ".jpg", ".jpeg", ".svg", ".csv", ".json", ".txt", ".html", ".md"):
                        generated_files.append(f)

            success = proc.returncode == 0
            error_summary = ""
            if not success:
                # Extract last meaningful error line
                err_lines = stderr.strip().split("\n")
                error_summary = err_lines[-1] if err_lines else "Unknown error"

            return ExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                duration_seconds=round(duration, 2),
                generated_files=generated_files,
                error_summary=error_summary,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start
            return ExecutionResult(
                success=False,
                error_summary=f"Execution timed out after {timeout} seconds",
                exit_code=-1,
                duration_seconds=round(duration, 2),
            )
        except Exception as e:
            duration = time.time() - start
            return ExecutionResult(
                success=False,
                error_summary=f"Execution failed: {type(e).__name__}: {e}",
                exit_code=-1,
                duration_seconds=round(duration, 2),
            )
        finally:
            # Clean up temp script (keep output_dir for results)
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass


def _docker_mount_path(path: str) -> str:
    """Normalize host path for Docker volume mounting across Windows and Linux."""
    resolved = os.path.realpath(path)
    return resolved.replace("\\", "/")


class DockerExecutor(ExecutorInterface):
    """Phase 2 executor: executes generated Python code inside an isolated Docker container.
    
    Security & Sandboxing controls:
    - Disposable container (--rm)
    - Runs as non-root user (--user 1000:1000)
    - Network isolation (--network none)
    - CPU limit (--cpus)
    - Memory limit (--memory)
    - Process count limit (--pids-limit)
    - Dropped capabilities (--cap-drop ALL)
    - Security option (--security-opt no-new-privileges:true)
    - Read-only data and script mounts
    - Secret hygiene (zero host env vars or API keys passed to container)
    - Strict timeout enforcement with automatic container cleanup
    """

    def __init__(
        self,
        image: str | None = None,
        cpus: float | None = None,
        memory: str | None = None,
        pids_limit: int | None = None,
        user: str | None = None,
    ):
        self.image = image or os.getenv("DOCKER_SANDBOX_IMAGE", "data-analyst-sandbox:latest")
        self.cpus = float(cpus if cpus is not None else os.getenv("DOCKER_MAX_CPUS", "2.0"))
        self.memory = memory or os.getenv("DOCKER_MAX_MEMORY", "512m")
        self.pids_limit = int(pids_limit if pids_limit is not None else os.getenv("DOCKER_PIDS_LIMIT", "64"))
        self.user = user or os.getenv("DOCKER_USER", "1000:1000")

    @property
    def backend_name(self) -> str:
        return "Docker Container Sandbox (Phase 2 — Isolated)"

    @property
    def is_sandboxed(self) -> bool:
        return True

    @staticmethod
    def is_docker_available() -> bool:
        """Check if Docker CLI is present and the Docker daemon is responding."""
        try:
            proc = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return proc.returncode == 0
        except Exception:
            return False

    def build_docker_command(
        self,
        container_name: str,
        script_host_dir: str,
        script_filename: str,
        dataset_host_dir: str,
        dataset_filename: str,
        output_host_dir: str,
    ) -> list[str]:
        """Construct the secure docker run command with full isolation constraints."""
        script_mount = f"{_docker_mount_path(script_host_dir)}:/workspace/script:ro"
        data_mount = f"{_docker_mount_path(dataset_host_dir)}:/workspace/data:ro"
        output_mount = f"{_docker_mount_path(output_host_dir)}:/workspace/output:rw"

        return [
            "docker", "run",
            "--name", container_name,
            "--rm",
            "--network", "none",
            "--user", self.user,
            "--cpus", str(self.cpus),
            "--memory", self.memory,
            "--pids-limit", str(self.pids_limit),
            "--security-opt", "no-new-privileges:true",
            "--cap-drop", "ALL",
            "-v", script_mount,
            "-v", data_mount,
            "-v", output_mount,
            "-w", "/workspace/output",
            self.image,
            "python", f"/workspace/script/{script_filename}",
        ]

    def execute(
        self,
        code: str,
        dataset_path: str,
        output_dir: str,
        timeout: int = 120,
    ) -> ExecutionResult:
        """Execute Python code in an isolated Docker container with resource constraints."""
        start = time.time()

        # Validate dataset existence
        dataset_path = os.path.realpath(dataset_path)
        output_dir = os.path.realpath(output_dir)

        if not os.path.isfile(dataset_path):
            return ExecutionResult(
                success=False,
                error_summary=f"Dataset not found: {dataset_path}",
                exit_code=-1,
                duration_seconds=0,
            )

        os.makedirs(output_dir, exist_ok=True)

        # Defense-in-depth static safety validation
        safety_warnings = validate_code_safety(code)
        if any("Blocked" in w for w in safety_warnings):
            return ExecutionResult(
                success=False,
                error_summary=f"Code rejected: {'; '.join(safety_warnings)}",
                exit_code=-1,
                duration_seconds=0,
            )

        # Verify Docker daemon availability explicitly
        if not self.is_docker_available():
            duration = round(time.time() - start, 2)
            msg = (
                "Docker daemon is unavailable or not running. "
                "Phase 2 isolated Docker sandboxing cannot proceed without an active Docker daemon. "
                "Execution was NOT run on host to preserve isolation boundaries."
            )
            logger.error(msg)
            return ExecutionResult(
                success=False,
                error_summary=msg,
                exit_code=-1,
                duration_seconds=duration,
            )

        # Create temporary script directory on host for mounting
        tmp_dir = tempfile.mkdtemp(prefix="analyst_docker_")
        dataset_filename = os.path.basename(dataset_path)
        dataset_host_dir = os.path.dirname(dataset_path)

        # Container internal paths
        container_dataset_path = f"/workspace/data/{dataset_filename}"
        container_output_dir = "/workspace/output"

        # Inject container-mapped paths
        header = (
            f"DATASET_PATH = {repr(container_dataset_path)}\n"
            f"OUTPUT_DIR = {repr(container_output_dir)}\n\n"
        )
        full_code = header + code

        script_filename = "analysis_script.py"
        script_host_path = os.path.join(tmp_dir, script_filename)
        container_name = f"analyst_sandbox_{uuid.uuid4().hex[:12]}"

        try:
            with open(script_host_path, "w", encoding="utf-8") as f:
                f.write(full_code)

            cmd = self.build_docker_command(
                container_name=container_name,
                script_host_dir=tmp_dir,
                script_filename=script_filename,
                dataset_host_dir=dataset_host_dir,
                dataset_filename=dataset_filename,
                output_host_dir=output_dir,
            )

            logger.info("Executing container: %s (image=%s, cpus=%s, memory=%s)", container_name, self.image, self.cpus, self.memory)

            # Clean host environment: do NOT pass NVIDIA_API_KEY or any secrets
            clean_env = {
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            }

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=clean_env,
            )

            duration = round(time.time() - start, 2)
            stdout = proc.stdout[:MAX_STDOUT] if proc.stdout else ""
            stderr = proc.stderr[:MAX_STDERR] if proc.stderr else ""

            # Collect generated files from output directory
            generated_files = []
            if os.path.isdir(output_dir):
                for f in os.listdir(output_dir):
                    fpath = os.path.join(output_dir, f)
                    if os.path.isfile(fpath):
                        ext = os.path.splitext(f)[1].lower()
                        if ext in (".png", ".jpg", ".jpeg", ".svg", ".csv", ".json", ".txt", ".html", ".md"):
                            generated_files.append(f)

            success = proc.returncode == 0
            error_summary = ""
            if not success:
                err_lines = stderr.strip().split("\n")
                error_summary = err_lines[-1] if err_lines else f"Container exited with code {proc.returncode}"

            return ExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                duration_seconds=duration,
                generated_files=generated_files,
                error_summary=error_summary,
            )

        except subprocess.TimeoutExpired:
            duration = round(time.time() - start, 2)
            try:
                subprocess.run(["docker", "kill", container_name], capture_output=True, timeout=5)
            except Exception:
                pass
            return ExecutionResult(
                success=False,
                error_summary=f"Execution timed out after {timeout} seconds in Docker container",
                exit_code=-1,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = round(time.time() - start, 2)
            return ExecutionResult(
                success=False,
                error_summary=f"Docker execution failed: {type(e).__name__}: {e}",
                exit_code=-1,
                duration_seconds=duration,
            )
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
            try:
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=5)
            except Exception:
                pass


class DemoExecutor(ExecutorInterface):
    """Deterministic executor for demonstration mode (no LLM required).
    
    Runs trusted, pre-defined analysis operations.
    Clearly labeled as demonstration — not LLM-generated.
    """

    def __init__(self, use_docker: bool = False):
        self.use_docker = use_docker

    @property
    def backend_name(self) -> str:
        if self.use_docker and DockerExecutor.is_docker_available():
            return "Deterministic Demo (Docker Sandboxed)"
        return "Deterministic Demo (Local Subprocess)"

    @property
    def is_sandboxed(self) -> bool:
        # Never claim sandboxed unless actually running in an active Docker container
        return self.use_docker and DockerExecutor.is_docker_available()

    def execute(
        self,
        code: str,
        dataset_path: str,
        output_dir: str,
        timeout: int = 120,
    ) -> ExecutionResult:
        """Execute the provided deterministic analysis code."""
        if self.use_docker and DockerExecutor.is_docker_available():
            executor: ExecutorInterface = DockerExecutor()
        else:
            executor = SubprocessExecutor()
        return executor.execute(code, dataset_path, output_dir, timeout)


def get_executor(
    is_demo: bool = False,
    backend: str | None = None,
) -> ExecutorInterface:
    """Factory: return the appropriate executor based on configuration.
    
    Backend resolution:
    - If `backend` argument is specified ('docker', 'subprocess', 'auto'), it is respected.
    - Otherwise inspects EXECUTION_BACKEND environment variable.
    - If `is_demo`: returns DemoExecutor.
    - If backend == 'docker': returns DockerExecutor (explicit failure if Docker is down).
    - If backend == 'subprocess': returns SubprocessExecutor.
    - If backend == 'auto' (default): returns DockerExecutor if Docker is available, else SubprocessExecutor.
    """
    configured_backend = (backend or os.getenv("EXECUTION_BACKEND", "auto")).lower()

    if is_demo:
        use_docker = configured_backend == "docker" or (configured_backend == "auto" and DockerExecutor.is_docker_available())
        return DemoExecutor(use_docker=use_docker)

    if configured_backend == "docker":
        return DockerExecutor()

    if configured_backend == "subprocess":
        return SubprocessExecutor()

    # 'auto' mode
    if DockerExecutor.is_docker_available():
        return DockerExecutor()

    return SubprocessExecutor()


def code_hash(code: str) -> str:
    """Hash code to detect repeated identical submissions."""
    return hashlib.sha256(code.encode()).hexdigest()[:16]

