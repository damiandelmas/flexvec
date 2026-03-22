"""python -m flexvec download-model"""
import sys

if len(sys.argv) > 1 and sys.argv[1] == "download-model":
    from flexvec.onnx.fetch import download_model
    download_model()
else:
    print("usage: python -m flexvec download-model")
