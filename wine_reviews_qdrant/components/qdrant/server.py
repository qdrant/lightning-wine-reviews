import subprocess
import time
from pathlib import Path
from typing import Text, Optional, List, Union

import lightning as L
from lightning_app import BuildConfig
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException


class QdrantServerComponent(L.LightningWork):
    """
    A component create to allow running Qdrant server, a neural search engine,
    in a Lightning app, as one of its components.
    """

    def __init__(
        self,
        version: Text = "latest",
        http_port: int = 6333,
        grpc_port: int = 6334,
        start_timeout: int = 30,
        volume_location: Optional[Union[Text, Path]] = None,
    ) -> None:
        image_name = f"qdrant/qdrant:{version}"
        super().__init__(
            parallel=True,
            local_build_config=BuildConfig(image=image_name),
            cloud_build_config=BuildConfig(image=image_name),
        )

        self.image_name = image_name
        self.http_port = http_port
        self.grpc_port = grpc_port
        self.start_timeout = start_timeout
        self.volume_location = str(volume_location)
        self.server_running = False
        self._process = None

    def run(self):
        if not self._is_docker_installed():
            raise RuntimeError(
                "Qdrant server has to be launched with Docker. "
                "Please make sure Docker executables are available."
            )

        qdrant_run_command = self.run_command()
        self._process = subprocess.Popen(qdrant_run_command)

        # Wait the selected timeout
        start = time.monotonic()
        while start - time.monotonic() < self.start_timeout:
            try:
                # Check if Qdrant server is up
                qdrant_client = QdrantClient(port=self.http_port)
                qdrant_client.openapi_client.cluster_api.cluster_status()
                self.server_running = True
                break
            except ResponseHandlingException:
                time.sleep(1)

    def on_exit(self):
        if self._process:
            self._process.terminate()

    def run_command(self) -> List[Text]:
        qdrant_run_command = [
            "docker",
            "run",
            "-it",
        ]
        if self.http_port is not None:
            qdrant_run_command += ["-p", f"{self.http_port}:6333"]
        if self.grpc_port is not None:
            qdrant_run_command += ["-p", f"{self.grpc_port}:6334"]
        if self.volume_location is not None:
            qdrant_run_command += ["-v", f"{self.volume_location}:/qdrant/storage"]
        qdrant_run_command += [self.image_name]
        return qdrant_run_command

    def _is_docker_installed(self) -> bool:
        try:
            process = subprocess.Popen(["docker", "--version"])
            return 0 == process.wait(timeout=1)
        except FileNotFoundError:
            # That means the docker executable has not been found
            return False
