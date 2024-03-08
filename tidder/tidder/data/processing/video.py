import os
import tempfile

import attrs
import requests
from loguru import logger

from tidder.data.videos import Video

from .base import Processor


@attrs.define(kw_only=True)
class VideoProcessor(Processor[Video]):
    video_info_file_name: str
    subtitle_file_name: str
    fallback_subtitle_file_name: str

    # TODO: Test this function
    def _process_remote_document(self, file_url: str) -> Video:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Get the file name from the URL
            file_name = file_url.split("/")[-1]

            # Send a HTTP request to the URL
            response = requests.get(file_url, stream=True)

            # Check if the request was successful
            if response.status_code == 200:
                # Write the contents of the response to a file in the temporary directory
                with open(os.path.join(tmp_dir, file_name), "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)

                # Process the downloaded file
                return self._process_local_document(os.path.join(tmp_dir, file_name))
            else:
                response.raise_for_status()

    def _process_local_document(self, file_path: str) -> Video:
        return self.load_video(file_path)

    def load_video(
        self,
        video_folder_path: str,
    ) -> Video:
        video_info_file = os.path.join(video_folder_path, self.video_info_file_name)
        try:
            video = Video.from_files(
                video_info_file,
                captions_file=os.path.join(video_folder_path, self.subtitle_file_name),
            )
        except FileNotFoundError:
            logger.warning(
                (
                    f"Could not find subtitles for {video_folder_path}, "
                    f"fallbacking to {self.fallback_subtitle_file_name}"
                )
            )
            try:
                video = Video.from_files(
                    video_info_file,
                    captions_file=os.path.join(
                        video_folder_path, self.fallback_subtitle_file_name
                    ),
                )
            except FileNotFoundError as err:
                logger.error(
                    f"Could not find any subtitle. Skipping video {video_folder_path}."
                )
                raise err

        return video
