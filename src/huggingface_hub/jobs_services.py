# -*- coding: utf-8 -*-
# ruff: noqa: F401
# Copyright 2026-present, HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Template services for HF Jobs.

This module provides pre-built templates for popular distributed computing
frameworks (Ray, Dask, Spark) that can be passed directly to `run_job()` or `run_uv_job()`
via the `with_services` parameter.

Example:
    ```python
    >>> from huggingface_hub import run_uv_job
    >>> from huggingface_hub.jobs_services import dask

    >>> job = run_uv_job(
    ...     script="your_dask_script.py,
    ...     with_services=dask(num_workers=2),
    ... )
    ```

See the individual functions for more details:
    - :func:`ray`: Ray cluster (head + workers)
    - :func:`dask`: Dask distributed cluster (scheduler + workers)
    - :func:`spark`: Spark standalone cluster (master + workers)
    - :func:`spark_connect`: Spark Connect standalone cluster (master + connect + workers)
"""

from __future__ import annotations

from typing import Any


def ray(
    *,
    num_workers: int = 2,
    image: str | None = None,
    ray_version: str | None = None,
    disable_usage_stats: bool = True,
    dashboard_host: str = "0.0.0.0",
    flavor: str = "cpu-upgrade",
    head_flavor: str = "cpu-upgrade",
) -> dict[str, Any]:
    """
    Create a Ray cluster services template.

    Sets up a Ray cluster with a head node and worker nodes. The head exposes the
    Ray Dashboard on port 8265. Workers join the cluster via the network group prefix.

    The main job should use `JobSubmissionClient` to submit work, as direct
    `ray.init(address=...)` is unavailable in HF Compute due to GCS IP advertisement.

    Services names:

    * `"ray-head"`
    * `"ray-worker"` (`num_workers` replicas, defaults to 2: `"ray-worker-0"` and `"ray-worker-1"`)

    Services are in the same network group, and are accessible via `${HF_NETWORK_GROUP_PREFIX}service-name:PORT`.

    Connect to `"ray-head"` using `JobSubmissionClient(f"http://{os.environ['HF_NETWORK_GROUP_PREFIX']}ray-head:8265")`

    The template is equivalent to this `ray-docker-compose.yml` file for HF Jobs services:

    ```yaml
    services:
        ray-head:
            image: python:3.12
            command: ["sh", "-c", "pip install -q 'ray[default]' && ray start --head --port=6379 --dashboard-host=0.0.0.0 --disable-usage-stats && sleep infinity"]
            env:
                RAY_DEDUP_LOGS: "false"
            flavor: "cpu-upgrade"

        ray-worker:
            image: python:3.12
            command: ["sh", "-c", "pip install -q 'ray[default]' && ray start --address=${HF_NETWORK_GROUP_PREFIX}ray-head:6379 --disable-usage-stats && sleep infinity"]
            replicas: 2
            flavor: "cpu-upgrade"
    ```

    Args:
        num_workers (`int`, *optional*):
            Number of worker nodes. Defaults to 2.
        image (`str`, *optional*):
            Docker image to use. Defaults to `"python:3.12"`.
        ray_version (`str`, *optional*):
            Ray version to install. If None, uses the latest.
        disable_usage_stats (`bool`, *optional*):
            Whether to disable Ray usage stats reporting. Defaults to True.
        dashboard_host (`str`, *optional*):
            Host to bind the Ray Dashboard to. Defaults to `"0.0.0.0"`.
        flavor (`str`, *optional*):
            Hardware flavor for Ray workers. Defaults to "cpu-upgrade".
        head_flavor (`str`, *optional*):
            Hardware flavor for Ray head. Defaults to "cpu-upgrade".

    Returns:
        A dict compatible with the `with_services` parameter of `run_job()` or `run_uv_job()`.

    Example:
        ```python
        >>> from huggingface_hub import run_uv_job
        >>> from huggingface_hub.jobs_services import ray

        >>> job = run_uv_job(
        ...     script="your_ray_script.py",
        ...     with_services=ray(num_workers=4),
        ... )
        ```

        Ray script example:

        ```python
        # /// script
        # dependencies = ["ray[default]"]
        # ///
        import time, socket, os
        from ray.job_submission import JobSubmissionClient
        import tempfile

        time.sleep(15)
        head_alias = os.environ.get("HF_NETWORK_GROUP_PREFIX", "") + "ray-head"
        head_ip = socket.gethostbyname(head_alias)
        client = JobSubmissionClient(f"http://{head_ip}:8265")

        print(f"Submitting to Ray cluster at {head_ip}")
        job_code = '''import time, ray, os

        print("Starting on node:", os.environ.get("RAY_NODE_IP", "unknown"))
        ray.init(address="auto")
        print("Cluster resources:", ray.cluster_resources())

        @ray.remote
        def compute(x):
            return x * x

        futures = [compute.remote(i) for i in range(10)]
        results = ray.get(futures)
        print("Results:", results)
        ray.shutdown()
        '''
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "job.py"), "w") as f:
            f.write(job_code)

        print("Submitting job...")
        job_id = client.submit_job(
            entrypoint="python job.py",
            runtime_env={"working_dir": tmpdir},
        )
        for _ in range(30):
            time.sleep(2)
            status = client.get_job_status(job_id)
            if status.value in ("COMPLETED", "SUCCEEDED"):
                print("Job completed!")
                break
            elif status.value == "FAILED":
                print("Job FAILED!")
                break
        logs = client.get_job_logs(job_id)
        print("Job logs:")
        print(logs)

        ```

    Note:
        The Ray dashboard is accessible at port 8265. You can verify the head node is
        ready by checking `http://<head-address>:8265/api/cluster_status`.
    """
    image = image or "python:3.12"
    ray_flag = "" if ray_version is None else f"=={ray_version}"
    ray_pkg = f"ray{ray_flag}[default]" if ray_flag else "ray[default]"
    usage_stats_flag = "--disable-usage-stats" if disable_usage_stats else ""

    head_cmd = (
        f"pip install -q {ray_pkg} && "
        f"ray start --head --port=6379 --dashboard-host={dashboard_host} "
        f"{usage_stats_flag} && "
        "sleep infinity"
    )

    worker_cmd = (
        f"pip install -q {ray_pkg} && "
        f"ray start --address=${{HF_NETWORK_GROUP_PREFIX}}ray-head:6379 "
        f"{usage_stats_flag} && "
        "sleep infinity"
    )

    return {
        "services": {
            "ray-head": {
                "image": image,
                "command": ["sh", "-c", head_cmd],
                "env": {
                    "RAY_DEDUP_LOGS": "false",
                },
                "flavor": head_flavor,
            },
            "ray-worker": {
                "image": image,
                "command": ["sh", "-c", worker_cmd],
                "flavor": flavor,
                "replicas": num_workers,
            },
        },
    }


def dask(
    *,
    num_workers: int = 2,
    image: str | None = None,
    dask_version: str | None = None,
    flavor: str = "cpu-upgrade",
    scheduler_flavor: str = "cpu-upgrade",
) -> dict[str, Any]:
    """
    Create a Dask distributed cluster services template.

    Sets up a Dask distributed cluster with a scheduler and worker nodes.
    Workers connect to the scheduler via the network group prefix.

    Services names:

    * `"dask-scheduler"`
    * `"dask-worker"` (`num_workers` replicas, defaults to 2: `"dask-worker-0"` and `"dask-worker-1"`)

    Services are in the same network group, and are accessible via `${HF_NETWORK_GROUP_PREFIX}service-name:PORT`.

    Connect to `"dask-scheduler"` using `Client(f"http://{os.environ['HF_NETWORK_GROUP_PREFIX']}dask-scheduler:8786")`

    The template is equivalent to this `dask-docker-compose.yml` file for HF Jobs services:

    ```yaml
    services:
        dask-scheduler:
            image: python:3.12
            command: ["sh", "-c", "pip install -q dask distributed pandas pyarrow && sleep 3 && dask scheduler --host 0.0.0.0"]
            flavor: "cpu-upgrade"

        dask-worker:
            image: python:3.12
            command: ["sh", "-c", "pip install -q dask distributed pandas pyarrow && sleep 5 && dask worker tcp://${HF_NETWORK_GROUP_PREFIX}dask-scheduler:8786"]
            replicas: 2
            flavor: "cpu-upgrade"
    ```

    Args:
        num_workers (`int`, *optional*):
            Number of worker nodes. Defaults to 2.
        image (`str`, *optional*):
            Docker image to use. Defaults to `"python:3.12"`.
        dask_version (`str`, *optional*):
            Dask/distributed version to install. If None, uses the latest.
        flavor (`str`, *optional*):
            Hardware flavor for Dask workers. Defaults to "cpu-upgrade".
        scheduler_flavor (`str`, *optional*):
            Hardware flavor for Dask scheduler. Defaults to "cpu-upgrade".

    Returns:
        A dict compatible with the `with_services` parameter of `run_job()` or `run_uv_job()`.

    Example:
        ```python
        >>> from huggingface_hub import run_uv_job
        >>> from huggingface_hub.jobs_services import dask

        >>> job = run_uv_job(
        ...     script="your_dask_script.py",
        ...     with_services=dask(num_workers=4),
        ... )
        ```

        Dask script example:

        ```python
        # /// script
        # dependencies = ["dask[distributed]"]
        # ///
        import os
        from dask.distributed import Client
        from dask import delayed
        import time

        def main():
            scheduler_addr = os.environ.get("HF_NETWORK_GROUP_PREFIX", "") + "dask-scheduler:8786"
            print(f"Connecting to scheduler at: {scheduler_addr}")
            client = Client(scheduler_addr)
            print(f"Dask dashboard: {client.dashboard_link}")

            @delayed
            def compute(x):
                return x * x

            results = [compute(i) for i in range(10)]
            print("Results:", [r.compute() for r in results])
            print("Dask job completed!")

        if __name__ == "__main__":
            main()
        ```
    """
    image = image or "python:3.12"
    version_flag = f"=={dask_version}" if dask_version else ""
    dask_pkg = f"dask{version_flag} distributed{version_flag} pandas pyarrow"

    scheduler_cmd = f"pip install -q {dask_pkg} && sleep 3 && dask scheduler --host 0.0.0.0"
    worker_cmd = (
        f"pip install -q {dask_pkg} && sleep 5 && dask worker tcp://${{HF_NETWORK_GROUP_PREFIX}}dask-scheduler:8786"
    )

    return {
        "services": {
            "dask-scheduler": {
                "image": image,
                "command": ["sh", "-c", scheduler_cmd],
                "flavor": scheduler_flavor,
            },
            "dask-worker": {
                "image": image,
                "command": ["sh", "-c", worker_cmd],
                "flavor": flavor,
                "replicas": num_workers,
            },
        },
    }


def spark(
    *,
    num_workers: int = 2,
    spark_version: str = "3.5.0",
    flavor: str = "cpu-upgrade",
    master_flavor: str = "cpu-upgrade",
) -> dict[str, Any]:
    """
    Create a Spark standalone cluster services template.

    Sets up a Spark standalone cluster with a master and worker nodes.
    Uses the official Apache Spark Docker image.

    The main job should use `spark-submit` with `--master spark://${HF_NETWORK_GROUP_PREFIX}spark-master:7077`.

    Unlike the `spark_connect` services template, the Spark Job can not run with `run_uv_jobs` out-of-the-box
    since the Job needs JAVA to be properly installed.

    Services names:

    * `"spark-master"`
    * `"spark-worker"` (`num_workers` replicas, defaults to 2: `"spark-worker-0"` and `"spark-worker-1"`)

    Services are in the same network group, and are accessible via `${HF_NETWORK_GROUP_PREFIX}service-name:PORT`.

    Connect to `"spark-master"` using `f"spark://${HF_NETWORK_GROUP_PREFIX}spark-master:7077"`

    The template is equivalent to this `spark-connect-docker-compose.yml` file for HF Jobs services:

    ```yaml
    services:
        spark-master:
            image: "apache/spark:3.5.0"
            command: [
                "sh", "-c",
                "/opt/spark/sbin/start-master.sh && sleep 5 && /opt/spark/sbin/start-thriftserver.sh --master spark://$(hostname -i):7077 --driver-memory 2g --executor-memory 1g --conf spark.driver.host=$(hostname -i) > /dev/null 2>&1 & tail -f /dev/null"
            ]
            flavor: "cpu-upgrade"

        spark-worker:
            image: "apache/spark:3.5.0"
            command: [
                "sh", "-c",
                "/opt/spark/sbin/start-worker.sh spark://${HF_NETWORK_GROUP_PREFIX}spark-master:7077 && tail -f /dev/null"
            ]
            replicas: 2
            flavor: "cpu-upgrade"
    ```


    Args:
        num_workers (`int`, *optional*):
            Number of worker nodes. Defaults to 2.
        spark_version (`str`, *optional*):
            Spark version to use (must be available on Docker Hub). Defaults to "3.5.0".
        flavor (`str`, *optional*):
            Hardware flavor for Spark workers. Defaults to "cpu-upgrade".
        master_flavor (`str`, *optional*):
            Hardware flavor for Spark master. Defaults to "cpu-upgrade".

    Returns:
        A dict compatible with the `with_services` parameter of `run_job()` or `run_uv_job()`.

    Example:
        ```python
        >>> from huggingface_hub import run_job
        >>> from huggingface_hub.jobs_services import spark

        >>> with open("your_spark_script.py") as f:
        ...     spark_script_content = f.read()
        >>> job = run_job(
        ...     image="apache/spark:3.5.0",
        ...     command=["sh", "-c", 'cat > /tmp/wordcount.py << "PYEOF"\n'
        ...         + spark_script_content
        ...         + '\nPYEOF
        ...         export DRIVER_HOST=$(hostname -i)
        ...         /opt/spark/bin/spark-submit \
        ...             --master spark://${HF_NETWORK_GROUP_PREFIX}spark-master:7077 \
        ...             --deploy-mode client \
        ...             --conf spark.driver.host=$DRIVER_HOST \
        ...             /tmp/wordcount.py'
        ...     ],
        ...     with_services=spark(num_workers=4),
        ... )
        ```

        Spark script example:

        ```python
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("WordCount").getOrCreate()
        lines = spark.sparkContext.parallelize([
            "hello world",
            "hello spark",
            "world of spark",
            "hello world of spark",
        ])
        counts = lines.flatMap(lambda x: x.split(" ")).map(lambda w: (w, 1)).reduceByKey(lambda a, b: a + b)
        print("Word counts:", counts.collect())
        ```
    """
    image = f"apache/spark:{spark_version}"
    spark_master_cmd = (
        "/opt/spark/sbin/start-master.sh && "
        "sleep 5 && "
        "/opt/spark/sbin/start-thriftserver.sh --master spark://$(hostname -i):7077 "
        "--driver-memory 2g --executor-memory 1g "
        "--conf spark.driver.host=$(hostname -i) > /dev/null 2>&1 & "
        "tail -f /dev/null"
    )

    spark_worker_cmd = (
        "/opt/spark/sbin/start-worker.sh spark://${HF_NETWORK_GROUP_PREFIX}spark-master:7077 && tail -f /dev/null"
    )

    return {
        "services": {
            "spark-master": {
                "image": image,
                "command": ["sh", "-c", spark_master_cmd],
                "flavor": master_flavor,
            },
            "spark-worker": {
                "image": image,
                "command": ["sh", "-c", spark_worker_cmd],
                "flavor": flavor,
                "replicas": num_workers,
            },
        },
    }


def spark_connect(
    *,
    num_workers: int = 2,
    spark_version: str = "4.0.4",
    flavor: str = "cpu-upgrade",
    master_flavor: str = "cpu-upgrade",
    connect_flavor: str = "cpu-upgrade",
) -> dict[str, Any]:
    """
    Create a Spark standalone cluster services template with Spark Connect.

    Sets up a Spark standalone cluster with a master and worker nodes, and a Spark Connect node.
    Uses the official Apache Spark Docker image.

    The main job should use Spark Connect with `"sc://{HF_NETWORK_GROUP_PREFIX}spark-connect:15002"`.

    Unlike the `spark` services template, the Spark Job can run with `run_uv_jobs` since the Job
    doesn't need JAVA to be installed.

    Services names:

    * `"spark-master"`
    * `"spark-connect"`
    * `"spark-worker"` (`num_workers` replicas, defaults to 2: `"spark-worker-0"` and `"spark-worker-1"`)

    Services are in the same network group, and are accessible via `${HF_NETWORK_GROUP_PREFIX}service-name:PORT`.

    Connect to `"spark-connect"` using `SparkSession.builder.remote(f"sc://{os.environ['HF_NETWORK_GROUP_PREFIX']}spark-connect:15002")`

    The template is equivalent to this `sparkc-connect-docker-compose.yml` file for HF Jobs services:

    ```yaml
    services:
        spark-master:
            image: "apache/spark:4.0.4-python3"
            command: ["sh", "-c", "/opt/spark/sbin/start-master.sh && tail -f /dev/null"]
            flavor: "cpu-upgrade"

        spark-worker:
            image: "apache/spark:4.0.4-python3"
            command: [
                "sh", "-c",
                "/opt/spark/sbin/start-worker.sh spark://${HF_NETWORK_GROUP_PREFIX}spark-master:7077 && tail -f /dev/null"
            ]
            replicas: 2
            flavor: "cpu-upgrade"

        spark-connect:
            image: "apache/spark:4.0.4-python3"
            # Connect server submits to master, which distributes to workers
            command: [
                "sh", "-c",
                "SPARK_MASTER_URL=spark://${HF_NETWORK_GROUP_PREFIX}spark-master:7077 /opt/spark/sbin/start-connect-server.sh && tail -f /dev/null"
            ]
            flavor: "cpu-upgrade"
    ```

    Args:
        num_workers (`int`, *optional*):
            Number of worker nodes. Defaults to 2.
        spark_version (`str`, *optional*):
            Spark version to use (must be available on Docker Hub). Defaults to "4.0.4".
        flavor (`str`, *optional*):
            Hardware flavor for Spark workers. Defaults to "cpu-upgrade".
        master_flavor (`str`, *optional*):
            Hardware flavor for Spark master. Defaults to "cpu-upgrade".
        connect_flavor (`str`, *optional*):
            Hardware flavor for Spark Connect. Defaults to "cpu-upgrade".

    Returns:
        A dict compatible with the `with_services` parameter of `run_job()` or `run_uv_job()`.

    Example:
        ```python
        >>> from huggingface_hub import run_uv_job
        >>> from huggingface_hub.jobs_services import spark_connect

        >>> job = run_uv_job(
        ...     script="your_spark_connect_script.py",
        ...     with_services=spark_connect(num_workers=4),
        ... )
        ```

        Spark Connect script example:

        ```python
        # /// script
        # requires-python = ">=3.9"
        # dependencies = [
        #     "pyspark-client==4.0.4",
        # ]
        # ///
        import os
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import rand, col

        prefix = os.environ.get("HF_NETWORK_GROUP_PREFIX", "")
        spark_connect_host = prefix + "spark-connect"
        spark = SparkSession.builder \
            .remote(f"sc://{spark_connect_host}:15002") \
            .appName("SparkConnectExample") \
            .getOrCreate()

        N = 50_000_000
        print(f"Estimating Pi with {N} samples...")
        count = (
            spark.range(N)
            .withColumn("x", rand())
            .withColumn("y", rand())
            .filter((col("x") ** 2 + col("y") ** 2) < 1)
            .count()
        )
        pi_estimate = 4.0 * count / N
        print(f"Pi is approximately: {pi_estimate}")
        print("True Pi: 3.141592653589793")
        print(f"Error: {abs(pi_estimate - 3.141592653589793):.6f}")
        ```
    """
    image = f"apache/spark:{spark_version}-python3"
    spark_master_cmd = "/opt/spark/sbin/start-master.sh && tail -f /dev/null"

    spark_worker_cmd = (
        "/opt/spark/sbin/start-worker.sh spark://${HF_NETWORK_GROUP_PREFIX}spark-master:7077 && tail -f /dev/null"
    )

    spark_connect_cmd = "SPARK_MASTER_URL=spark://${HF_NETWORK_GROUP_PREFIX}spark-master:7077 /opt/spark/sbin/start-connect-server.sh && tail -f /dev/null"

    return {
        "services": {
            "spark-master": {
                "image": image,
                "command": ["sh", "-c", spark_master_cmd],
                "flavor": master_flavor,
            },
            "spark-worker": {
                "image": image,
                "command": ["sh", "-c", spark_worker_cmd],
                "flavor": flavor,
                "replicas": num_workers,
            },
            "spark-connect": {
                "image": image,
                "command": ["sh", "-c", spark_connect_cmd],
                "flavor": connect_flavor,
            },
        },
    }


# Mapping of template names to functions for CLI lookup
SERVICES_TEMPLATES = {
    "ray": ray,
    "dask": dask,
    "spark": spark,
    "spark_connect": spark_connect,
}

__all__ = list(SERVICES_TEMPLATES) + ["SERVICES_TEMPLATES"]
