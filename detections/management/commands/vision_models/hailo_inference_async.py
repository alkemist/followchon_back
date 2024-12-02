from functools import partial

import numpy as np
from hailo_platform import (HEF, VDevice, FormatType)
from loguru import logger


class HailoAsyncInference:
    target = None

    def __init__(self, vdevice: VDevice, hef_path_all, hef_path_chons, batch_size=1, output_type='FLOAT32'):
        """
        Initialize the HailoAsyncInference class with the provided HEF model file path.

        Args:
            hef_path (str): Path to the HEF model file.
            batch_size (int): Batch size for inference.
            output_type (str): Format type of the output stream.
        """

        self.hef_all = HEF(hef_path_all)
        self.hef_chons = HEF(hef_path_chons)
        self.target = vdevice

        self.infer_model_all = self.target.create_infer_model(hef_path_all)
        self.infer_model_chons = self.target.create_infer_model(hef_path_chons)

        self.infer_model_all.set_batch_size(batch_size)
        self.infer_model_chons.set_batch_size(batch_size)

        self._set_input_output(output_type)
        self.input_vstream_info_all, self.output_vstream_info_all = self._get_vstream_info_all()
        self.input_vstream_info_chons, self.output_vstream_info_chons = self._get_vstream_info_chons()

        self.configured_infer_model_all = self.infer_model_all.configure()
        self.configured_infer_model_chons = self.infer_model_chons.configure()

        self.output_results_all = []
        self.output_results_chons = []

    def _set_input_output(self, output_type):
        """
        Set the input and output layer information for the HEF model.

        Args:
            output_type (str): Format type of the output stream.
        """
        input_format_type_all = self.hef_all.get_input_vstream_infos()[0].format.type
        self.infer_model_all.input().set_format_type(input_format_type_all)

        input_format_type_chons = self.hef_chons.get_input_vstream_infos()[0].format.type
        self.infer_model_chons.input().set_format_type(input_format_type_chons)

        logger.info('Outputs :')
        logger.info(self.infer_model_all.output_names)
        logger.info(self.infer_model_chons.output_names)

        for name in self.infer_model_all.output_names:
            self.infer_model_all.output(name).set_format_type(getattr(FormatType, output_type))

        for name in self.infer_model_chons.output_names:
            self.infer_model_chons.output(name).set_format_type(getattr(FormatType, output_type))

    def callback_all(self, completion_info, binding, frame):
        """
        Callback function for handling inference results.

        Args:
            completion_info: Information about the completion of the inference task.
            binding: Bindings object containing input and output buffers.
        """
        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            for name in self.infer_model_all.output_names:
                self.output_results_all.append(binding.output(name).get_buffer())

    def callback_chons(self, completion_info, binding, frame):
        """
        Callback function for handling inference results.

        Args:
            completion_info: Information about the completion of the inference task.
            binding: Bindings object containing input and output buffers.
        """
        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            for name in self.infer_model_chons.output_names:
                self.output_results_chons.append(binding.output(name).get_buffer())

    def remove_last_output_results_all(self):
        return self.output_results_all.pop()

    def remove_last_output_results_chons(self):
        return self.output_results_chons.pop()

    def _get_vstream_info_all(self):
        """
        Get information about input and output stream layers.

        Returns:
            tuple: List of input stream layer information, List of output stream layer information.
        """
        return self.hef_all.get_input_vstream_infos(), self.hef_all.get_output_vstream_infos()

    def _get_vstream_info_chons(self):
        """
        Get information about input and output stream layers.

        Returns:
            tuple: List of input stream layer information, List of output stream layer information.
        """
        return self.hef_chons.get_input_vstream_infos(), self.hef_chons.get_output_vstream_infos()

    def get_input_shape_all(self):
        """
        Get the shape of the model's input layer.

        Returns:
            tuple: Shape of the model's input layer.
        """
        return self.input_vstream_info_all[0].shape  # Assumes that the model has one input

    def get_input_shape_chons(self):
        """
        Get the shape of the model's input layer.

        Returns:
            tuple: Shape of the model's input layer.
        """
        return self.input_vstream_info_chons[0].shape  # Assumes that the model has one input

    def get_output_results_all(self):
        """
        Get the results of the inference.

        Returns:
            list: List of inference outputs.
        """
        return self.output_results_all

    def get_output_results_chons(self):
        """
        Get the results of the inference.

        Returns:
            list: List of inference outputs.
        """
        return self.output_results_chons

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

        job_all = None
        job_chons = None

        for frame in input_data:
            binding_all = self._create_bindings_all()
            binding_all.input().set_buffer(frame)

            self.configured_infer_model_all.wait_for_async_ready(timeout_ms=10000)
            job_all = self.configured_infer_model_all.run_async(
                [binding_all],
                partial(
                    self.callback_all, binding=binding_all, frame=frame
                )
            )

        if job_all is not None:
            job_all.wait(10000)  # Wait for the last job

        for frame in input_data:
            binding_chons = self._create_bindings_chons()
            binding_chons.input().set_buffer(frame)

            self.configured_infer_model_chons.wait_for_async_ready(timeout_ms=10000)
            job_chons = self.configured_infer_model_chons.run_async(
                [binding_chons],
                partial(
                    self.callback_chons, binding=binding_chons, frame=frame
                )
            )

        if job_chons is not None:
            job_chons.wait(10000)  # Wait for the last job

        return self.output_results_all + self.output_results_chons

    def _create_bindings_all(self):
        """
        Create bindings for input and output buffers.

        Returns:
            bindings: Bindings object with input and output buffers.
        """
        output_buffers = {
            name: np.empty(self.infer_model_all.output(name).shape, dtype=np.float32)
            for name in self.infer_model_all.output_names
        }

        return self.configured_infer_model_all.create_bindings(output_buffers=output_buffers)

    def _create_bindings_chons(self):
        """
        Create bindings for input and output buffers.

        Returns:
            bindings: Bindings object with input and output buffers.
        """
        output_buffers = {
            name: np.empty(self.infer_model_chons.output(name).shape, dtype=np.float32)
            for name in self.infer_model_chons.output_names
        }

        return self.configured_infer_model_chons.create_bindings(output_buffers=output_buffers)

    def release_device(self):
        """
        Release the Hailo device.
        """
        del self.configured_infer_model_all
        del self.configured_infer_model_chons
        self.target.release()
