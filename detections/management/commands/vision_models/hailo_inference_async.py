from functools import partial

import numpy as np
from hailo_platform import (HEF, VDevice, FormatType, HailoSchedulingAlgorithm)
from loguru import logger


class HailoAsyncInference:
    def __init__(self, hef_path, batch_size=1, output_type='FLOAT32'):
        """
        Initialize the HailoAsyncInference class with the provided HEF model file path.

        Args:
            hef_path (str): Path to the HEF model file.
            batch_size (int): Batch size for inference.
            output_type (str): Format type of the output stream.
        """
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)
        self._set_input_output(output_type)
        self.input_vstream_info, self.output_vstream_info = self._get_vstream_info()
        self.configured_infer_model = self.infer_model.configure()
        self.output_results = []

    def _set_input_output(self, output_type):
        """
        Set the input and output layer information for the HEF model.

        Args:
            output_type (str): Format type of the output stream.
        """
        input_format_type = self.hef.get_input_vstream_infos()[0].format.type
        self.infer_model.input().set_format_type(input_format_type)
        self.infer_model.output().set_format_type(getattr(FormatType, output_type))

    def callback(self, completion_info, bindings):
        """
        Callback function for handling inference results.

        Args:
            completion_info: Information about the completion of the inference task.
            bindings: Bindings object containing input and output buffers.
        """
        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            self.output_results.append(bindings.output().get_buffer()[0])

    def remove_last_output_results(self):
        return self.output_results.pop()

    def _get_vstream_info(self):
        """
        Get information about input and output stream layers.

        Returns:
            tuple: List of input stream layer information, List of output stream layer information.
        """
        return self.hef.get_input_vstream_infos(), self.hef.get_output_vstream_infos()

    def get_input_shape(self):
        """
        Get the shape of the model's input layer.

        Returns:
            tuple: Shape of the model's input layer.
        """
        return self.input_vstream_info[0].shape  # Assumes that the model has one input

    def get_output_results(self):
        """
        Get the results of the inference.

        Returns:
            list: List of inference outputs.
        """
        return self.output_results

    def run(self, input_data):
        """
        Run asynchronous inference on the Hailo-8 device.

        Args:
            input_data (np.ndarray): Input data for inference.

        Returns:
            list: List of inference outputs.
        """
        if input_data.ndim == 1 or input_data.size == 0 or input_data is None:
            logger.error('Input data is empty')
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)

        job = None

        for frame in input_data:
            bindings = self._create_bindings()
            bindings.input().set_buffer(frame)
            self.configured_infer_model.wait_for_async_ready(timeout_ms=10000)
            job = self.configured_infer_model.run_async([bindings], partial(self.callback, bindings=bindings))

        if job is not None:
            job.wait(10000)  # Wait for the last job

        return self.output_results

    def _create_bindings(self):
        """
        Create bindings for input and output buffers.

        Returns:
            bindings: Bindings object with input and output buffers.
        """
        output_buffers = {name: np.empty(self.infer_model.output(name).shape, dtype=np.float32)
                          for name in self.infer_model.output_names}
        return self.configured_infer_model.create_bindings(output_buffers=output_buffers)

    def release_device(self):
        """
        Release the Hailo device.
        """
        del self.configured_infer_model
        self.target.release()
