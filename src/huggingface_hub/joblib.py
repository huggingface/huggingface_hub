import ast
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Optional, TypeVar, Union

from ._space_api import SpaceHardware
from .hf_api import HfApi


try:
    import cloudpickle
    from joblib import ParallelBackendBase as _ParallelBackendBase
    from joblib import register_parallel_backend as _register_parallel_backend
    from joblib.parallel import BatchedCalls as _BatchedCalls
except ImportError:
    cloudpickle = None
    _ParallelBackendBase = object
    _register_parallel_backend = None
    _BatchedCalls = None


T = TypeVar("T")
C = TypeVar("C", bound=Callable)

_BASE_DEPENDENCIES = ["joblib", "cloudpickle", "huggingface_hub"]
_RESULT_SENTINEL = "=== HF JOBS RESULT ==="
_ERROR_SENTINEL = "=== HF JOBS ERROR ==="
_UV_SCRIPT_TEMPLATE = """import cloudpickle

print("Running {func_name}")
func = cloudpickle.loads({cloudpickled_func})
try:
    result = func()
except Exception as err:
    print("{error_sentinel}")
    print(cloudpickle.dumps(err))
    exit(1)
else:
    print("{result_sentinel}")
    print(cloudpickle.dumps(result))
"""


class HfJobsError(RuntimeError):
    pass


class HfJobsBackend(_ParallelBackendBase):
    default_n_jobs = -1
    MAX_JOBS_IN_PARALLEL = 100
    supports_retrieve_callback = True

    def __init__(
        self,
        dependencies: Optional[list[str]] = None,
        python: Optional[str] = None,
        image: Optional[str] = None,
        env: Optional[dict[str, Any]] = None,
        secrets: Optional[dict[str, Any]] = None,
        flavor: Optional[SpaceHardware] = None,
        timeout: Optional[Union[int, float, str]] = None,
        labels: Optional[dict[str, str]] = None,
        namespace: Optional[str] = None,
        token: Union[bool, str, None] = None,
        **backend_kwargs,
    ):
        self.dependencies = dependencies
        self.python = python
        self.image = image
        self.env = env
        self.secrets = secrets
        self.flavor = flavor
        self.timeout = timeout
        self.labels = labels
        self.namespace = namespace
        self.token = token
        super().__init__(**backend_kwargs)
        self.n_tasks = None

    def configure(self, n_jobs=None, parallel=None, **backend_kwargs):
        """Configure the backend for a specific instance of Parallel."""
        for key, value in backend_kwargs.items():
            setattr(self, key, value)
        n_jobs = self.effective_n_jobs(n_jobs)
        self._executor = ThreadPoolExecutor(n_jobs)

        # Return the effective number of jobs
        return n_jobs

    def terminate(self):
        """Clean-up the resources associated with the backend."""
        self._executor.shutdown()
        self._executor = None

    def effective_n_jobs(self, n_jobs):
        """Determine the number of jobs that can be run in parallel."""
        if n_jobs is None or n_jobs < 0:
            if self.n_tasks is not None:
                n_jobs = self.n_tasks
            else:
                n_jobs = self.MAX_JOBS_IN_PARALLEL
        # we never set 1 here or joblib uses local sequential output
        return max(2, n_jobs)

    def submit(self, func, callback):
        """Schedule a function to be run and return a future-like object.

        This method should return a future-like object that allow tracking
        the progress of the task.

        If ``supports_retrieve_callback`` is False, the return value of this
        method is passed to ``retrieve_result`` instead of calling
        ``retrieve_result_callback``.

        Parameters
        ----------
        func: callable
            The function to be run in parallel.

        callback: callable
            A callable that will be called when the task is completed. This callable
            is a wrapper around ``retrieve_result_callback``. This should be added
            to the future-like object returned by this method, so that the callback
            is called when the task is completed.

            For future-like backends, this can be achieved with something like
            ``future.add_done_callback(callback)``.

        Returns
        -------
        future: future-like
            A future-like object to track the execution of the submitted function.
        """
        hf_jobs_func = partial(
            _hf_jobs_func,
            func,
            dependencies=self.dependencies,
            python=self.python,
            image=self.image,
            env=self.env,
            secrets=self.secrets,
            flavor=self.flavor,
            timeout=self.timeout,
            labels=self.labels,
            namespace=self.namespace,
            token=self.token,
        )
        future = self._executor.submit(hf_jobs_func)
        future.add_done_callback(callback)
        return future

    def retrieve_result_callback(self, future):
        """Called within the callback function passed to `submit`.

        This method can customise how the result of the function is retrieved
        from the future-like object.

        Parameters
        ----------
        future: future-like
            The future-like object returned by the `submit` method.

        Returns
        -------
        result: object
            The result of the function executed in parallel.
        """
        return future.result()


def _hf_jobs_func(func: Callable[..., T], namespace: Optional[str], token: Optional[str], **kwargs) -> T:
    if isinstance(func, _BatchedCalls):
        func_name = ", ".join({item[0].__name__: None for item in func.items})
    else:
        func_name = func.__name__
    code = _UV_SCRIPT_TEMPLATE.format(
        func_name=func_name,
        cloudpickled_func=cloudpickle.dumps(func),
        result_sentinel=_RESULT_SENTINEL,
        error_sentinel=_ERROR_SENTINEL,
    )
    dependencies = {}
    for dependency in _BASE_DEPENDENCIES + (kwargs.get("dependencies") or []):
        package_name = re.split(r"[^-\w]+", dependency)[0].replace("-", "_").lower()
        dependencies[package_name] = dependency
    kwargs["dependencies"] = list(dependencies.values())
    kwargs["labels"] = {"joblib": "", "func_name": func_name, **(kwargs.get("labels") or {})}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete_on_close=False) as tmp_f:
        tmp_f.write(code)
        tmp_f.close()
        api = HfApi(token=token)
        job = api.run_uv_job(tmp_f.name, namespace=namespace, **kwargs)
        regular_logs: list[str] = []
        result_logs: list[str] = []
        error_logs: list[str] = []
        current_logs = regular_logs
        for log in api.fetch_job_logs(job_id=job.id, namespace=namespace, follow=True):
            if log == _RESULT_SENTINEL:
                current_logs = result_logs
            elif log == _ERROR_SENTINEL:
                current_logs = error_logs
            else:
                current_logs.append(log)
        if result_logs:
            result_log = result_logs[-1]
            if not result_log.startswith("b'") and result_log.endwith("'"):
                raise ValueError(f"Bad result log fond after result sentinel: {result_log}")
            return cloudpickle.loads(ast.literal_eval(result_log))
        elif error_logs:
            result_log = result_logs[-1]
            if not result_log.startswith("b'") and result_log.endwith("'"):
                raise ValueError(f"Bad result log fond after result sentinel: {result_log}")
            err = cloudpickle.loads(ast.literal_eval(result_log))
            raise HfJobsError(f"Job {job.id} failed. See logs at {job.url}") from err
        else:
            raise ValueError(
                f"Failed to find result and result sentinel at the end of logs:\n\n...\n{'\n'.join(regular_logs[-10:])}"
            )


def register_hf_jobs() -> None:
    if _register_parallel_backend and cloudpickle:
        _register_parallel_backend("hf-jobs", HfJobsBackend)
    else:
        raise ImportError(
            "Please install `joblib` and `cloudpickle` to use the HF Jobs backend. You can install it with `pip install joblib cloudpickle`."
        )
