import unittest

import backend.transcription as t


class TestCudaRuntimeDetection(unittest.TestCase):
    def test_detects_missing_cudnn_runtime_message(self):
        err = RuntimeError("Could not locate cudnn_ops64_9.dll. Please make sure it is in your library path!")
        self.assertTrue(t._looks_like_missing_cuda_runtime(err))

    def test_detects_missing_cublas_runtime_message(self):
        err = RuntimeError("Library cublas64_12.dll is not found")
        self.assertTrue(t._looks_like_missing_cuda_runtime(err))


if __name__ == "__main__":
    unittest.main()
