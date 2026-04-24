import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRTViTFeatureExtractor:
    def __init__(self, engine_path, device="cuda", fixed_batch=500):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.device = device
        self.fixed_batch = fixed_batch

        # Allocate buffers for the fixed batch size
        self.inputs = []
        self.outputs = []
        self.bindings = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.context.get_tensor_shape(name)   # (500,3,224,224)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'host': host_mem, 'device': device_mem,
                                    'shape': shape, 'dtype': dtype})
            else:
                self.outputs.append({'name': name, 'host': host_mem, 'device': device_mem,
                                     'shape': shape, 'dtype': dtype})
        self.feat_dim = self.outputs[0]['shape'][1]
        self.input_dtype = self.inputs[0]['dtype']

    def extract(self, batch_tensor):
        """
        batch_tensor: torch Tensor (B,3,224,224) on CPU, B <= fixed_batch
        Returns: torch Tensor (B, feat_dim) on CPU
        """
        actual_batch = batch_tensor.shape[0]

        # Pad to fixed batch size if needed
        if actual_batch < self.fixed_batch:
            pad = torch.zeros(self.fixed_batch - actual_batch,
                              *batch_tensor.shape[1:], dtype=batch_tensor.dtype)
            batch_tensor = torch.cat([batch_tensor, pad], dim=0)

        # Copy to GPU
        batch_np = batch_tensor.numpy().astype(self.input_dtype)
        cuda.memcpy_htod(self.inputs[0]['device'], batch_np.ravel())

        # Run inference
        self.context.execute_v2(self.bindings)

        # Copy output back
        output = np.empty(self.outputs[0]['shape'], dtype=self.outputs[0]['dtype'])
        cuda.memcpy_dtoh(output, self.outputs[0]['device'])

        # Convert to torch and truncate to actual batch size
        embeddings = torch.from_numpy(output.astype(np.float32))
        return embeddings[:actual_batch]