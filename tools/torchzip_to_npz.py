"""Load a torch.save() zip checkpoint WITHOUT torch and save it as .npz.

Also understands the spaCy-transformers wheel layout (msgpack -> shim ->
{'config', 'state': <torch zip bytes>}), which lets us get real RoBERTa-base
weights from GitHub releases when huggingface.co is not reachable.

usage: python -I tools/torchzip_to_npz.py <file.whl|file.pt> out.npz
"""
import io, pickle, sys, zipfile
import numpy as np

DT = {"FloatStorage": np.float32, "HalfStorage": np.float16, "BFloat16Storage": np.uint16,
      "LongStorage": np.int64, "IntStorage": np.int32, "ByteStorage": np.uint8,
      "BoolStorage": np.bool_, "DoubleStorage": np.float64}


def load_torch_zip(buf):
    z = zipfile.ZipFile(io.BytesIO(buf))
    names = z.namelist()
    root = names[0].split("/")[0]

    class StorageType:
        def __init__(self, name): self.name = name

    def rebuild(storage, offset, size, stride, *a):
        arr = storage
        if not size:
            return arr[offset:offset + 1].reshape(())
        n = 1 + sum((s - 1) * st for s, st in zip(size, stride))
        flat = arr[offset:offset + n]
        return np.lib.stride_tricks.as_strided(
            flat, shape=size, strides=[st * arr.itemsize for st in stride]).copy()

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if mod == "torch" and name.endswith("Storage"):
                return StorageType(name)
            if mod == "collections" and name == "OrderedDict":
                import collections; return collections.OrderedDict
            if mod == "torch._utils" and name == "_rebuild_parameter":
                return lambda t, *a: t
            raise pickle.UnpicklingError(f"blocked {mod}.{name}")

        def persistent_load(self, pid):
            _, st, key, _loc, _n = pid
            dt = DT[st.name]
            raw = z.read(f"{root}/data/{key}")
            a = np.frombuffer(raw, dtype=dt)
            if st.name == "BFloat16Storage":
                a = (a.astype(np.uint32) << 16).view(np.float32)
            return a

    return U(io.BytesIO(z.read(f"{root}/data.pkl"))).load()


def main(src, out):
    if src.endswith(".whl"):
        import msgpack
        w = zipfile.ZipFile(src)
        name = [n for n in w.namelist() if n.endswith("transformer/model")][0]
        o = msgpack.unpackb(w.read(name), raw=False, strict_map_key=False)
        shim = [s for s in o["shims"] if s][0][0]
        buf = msgpack.unpackb(shim, raw=False)["state"]
    else:
        buf = open(src, "rb").read()
    sd = load_torch_zip(buf)
    sd = {k: np.asarray(v, dtype=np.float32) for k, v in sd.items() if hasattr(v, "dtype")}
    np.savez(out, **sd)
    for k, v in list(sd.items())[:40]:
        print(k, v.shape)
    print(len(sd), "tensors,", sum(v.size for v in sd.values()) / 1e6, "M params")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
