"""启动 torchrun，并兼容未编译 libuv 的 Windows PyTorch。"""
import os


def main():
    if os.name == "nt":
        import torch.distributed as distributed
        original_store = distributed.TCPStore

        class WindowsTCPStore(original_store):
            """保留原生 TCPStore 通信，只关闭 Windows 缺少的 libuv 后端。"""
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("use_libuv", False)
                super().__init__(*args, **kwargs)

        distributed.TCPStore = WindowsTCPStore
        os.environ.setdefault("USE_LIBUV", "0")
    from torch.distributed.run import main as torchrun_main
    torchrun_main()


if __name__ == "__main__":
    main()
