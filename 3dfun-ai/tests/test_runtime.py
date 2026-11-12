from engine.recon import runtime


def test_unavailable_gpu_uses_cpu(monkeypatch):
    monkeypatch.setenv("RECON_DEVICE", "cuda")
    monkeypatch.setattr(runtime, "_available", lambda: (runtime.CPU,))
    assert runtime.runtime_status() == {
        "requested": "cuda", "preferred": runtime.CPU, "available": [runtime.CPU],
    }


def test_gpu_failure_reopens_session_on_cpu(monkeypatch, tmp_path):
    opened = []

    class Session:
        def __init__(self, provider):
            self.provider = provider

        def get_providers(self):
            return [self.provider]

        def run(self, outputs, feeds):
            if self.provider == runtime.DIRECTML:
                raise RuntimeError("simulated device failure")
            return [feeds["image"]]

    def open_session(_self, provider):
        opened.append(provider)
        return Session(provider)

    monkeypatch.setattr(runtime.InferenceSession, "_open", open_session)
    session = runtime.InferenceSession(tmp_path / "model.onnx", None, runtime.DIRECTML)
    assert session.run(["output"], {"image": 42}) == [42]
    assert session.provider == runtime.CPU
    assert opened == [runtime.DIRECTML, runtime.CPU]
