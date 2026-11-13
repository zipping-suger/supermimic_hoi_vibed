"""CUDA-graph capture of fixed-shape training functions.

A ``CUDACall`` wraps one function whose tensor arguments keep their shapes. Arguments are copied
into stable buffers, the function runs eagerly for a few warm-up calls, is then captured once, and
afterwards every call is a graph replay. Replay sampling, admission checks and metric reads stay
on the host; the graph owns forward, backward and the gated optimizer kernels.

The learning graphs (world model, contact head, actor, critic) run strictly one after another, so
they share one scratch memory pool. Outputs therefore live outside the pool: a captured temporary
that escaped into another graph would be overwritten by the next capture or replay.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch
from torch.utils._pytree import tree_flatten, tree_map


class LearningGraphPool:
    """One scratch pool and capture stream for learners that run serially on one stream."""

    def __init__(self):
        self.device = torch.cuda.current_device()
        self.execution_stream = torch.cuda.current_stream().cuda_stream
        self.stream = torch.cuda.Stream()
        self.token = torch.cuda.graph_pool_handle()

    def check_stream(self):
        if torch.cuda.current_device() != self.device or torch.cuda.current_stream().cuda_stream != self.execution_stream:
            raise RuntimeError("Shared learning graphs must run serially on their original CUDA stream")


class TorchCaptureAllocator:
    """Serve Warp's temporary allocations from Torch during capture (CUDA 12.2 forbids async malloc there)."""

    def __init__(self, device: str):
        import warp as wp

        self.device, self.memory_kind, self.live = device, wp.MemoryKind.CUDA_DEVICE, {}

    def allocate(self, size_in_bytes):
        value = torch.empty(size_in_bytes, dtype=torch.uint8, device=self.device)
        self.live[value.data_ptr()] = value
        return value.data_ptr()

    def deallocate(self, ptr, size_in_bytes):
        self.live.pop(ptr, None)


def _storage(x):
    return x.device, x.untyped_storage().data_ptr()


class CUDACall:
    """Capture ``fn`` after ``warmup`` eager calls and replay it afterwards.

    simulation: the mjlab ``Simulation`` whose Warp kernels ``fn`` launches (collection). Its
        outputs are cloned on return because collection rows outlive the next replay.
    pool: shared ``LearningGraphPool``. Outputs are then exported to graph-external buffers,
        and learning consumes them before the next call.
    capture_after: a ``CUDACall`` that must be captured first; until then calls run eagerly, so
        the shared pool is filled in a fixed order.
    borrowed_inputs: argument positions passed by reference instead of copied. The caller must
        pass the same storage every time and may let ``fn`` overwrite it.
    """

    def __init__(self, fn, *, simulation=None, pool: LearningGraphPool | None = None, capture_after=None,
                 borrowed_inputs=(), warmup: int = 3):  # fmt: skip
        if pool is not None and simulation is not None:
            raise ValueError("Simulation capture uses its own pool")
        if borrowed_inputs and pool is None:
            raise ValueError("Borrowed inputs require the shared learning pool")
        self.fn, self.simulation, self.pool, self.capture_after = fn, simulation, pool, capture_after
        self.borrowed, self.warmup = frozenset(borrowed_inputs), warmup
        self.calls = self.replays = 0
        self.graph = self.inputs = self.out = None

    def _bind(self, args):
        """Copy ``args`` into the stable input buffers (created on the first call)."""
        if self.inputs is None:
            self.inputs = tuple(arg.detach() if i in self.borrowed else tree_map(lambda x: None if x is None else x.detach().clone(), arg)
                                for i, arg in enumerate(args))  # fmt: skip
            return
        for i in self.borrowed:
            if args[i].data_ptr() != self.inputs[i].data_ptr() or args[i].stride() != self.inputs[i].stride():
                raise ValueError("Borrowed CUDA graph input storage changed")
        source, spec = tree_flatten(args)
        target, expected = tree_flatten(self.inputs)
        same = lambda a, b: (a is None and b is None) or (  # noqa: E731
            a is not None and b is not None and a.shape == b.shape and a.dtype == b.dtype and a.device == b.device)
        if spec != expected or not all(same(a, b) for a, b in zip(source, target)):
            raise ValueError("CUDA graph inputs changed shape, dtype, device or structure")
        for a, b in zip(source, target):
            if a is not None and a.data_ptr() != b.data_ptr():
                b.copy_(a)

    def _export(self, out):
        """Copy outputs into graph-external buffers; views of the static inputs stay views."""
        source, spec = tree_flatten(out)
        if self.out is None:
            inputs, _ = tree_flatten(self.inputs)
            storage = {_storage(x) for x in inputs if x is not None}
            self._aliases = [_storage(x) in storage for x in source]
            self.out = tree_map(lambda x: x.detach() if _storage(x) in storage else x.detach().clone(), out)
            return self.out
        target, expected = tree_flatten(self.out)
        if spec != expected:
            raise ValueError("CUDA graph output structure changed")
        for a, b, alias in zip(source, target, self._aliases):
            if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                raise ValueError("CUDA graph output shape, dtype or device changed")
            if not alias:
                b.copy_(a.detach())
            elif a.data_ptr() != b.data_ptr() or a.stride() != b.stride():
                raise ValueError("CUDA graph output no longer aliases its static input")
        return self.out

    @contextmanager
    def _capture_warp(self):
        """Record the simulation's Warp kernels into the Torch graph being captured."""
        if self.simulation is None:
            yield
            return
        import warp as wp

        sim = self.simulation
        own_graphs, sim.use_cuda_graph = sim.use_cuda_graph, False  # launch kernels, not mjlab's own graphs
        try:
            stream = wp.stream_from_torch(torch.cuda.current_stream())
            self._warp_allocator = TorchCaptureAllocator(str(sim.wp_device))
            with wp.ScopedStream(stream, sync_enter=False, sync_exit=False), wp.ScopedAllocator(sim.wp_device, self._warp_allocator):
                wp.capture_begin(stream=stream, external=True)
                try:
                    yield
                finally:
                    self._warp_graph = wp.capture_end(stream=stream)  # keeps the captured kernel modules alive
        finally:
            sim.use_cuda_graph = own_graphs

    def __call__(self, *args):
        if self.pool is not None:
            self.pool.check_stream()
        self._bind(args)
        self.calls += 1
        external = self.pool is not None
        if self.calls <= self.warmup:
            begin = time.monotonic()
            if self.simulation is not None:
                out = self.fn(*self.inputs)
            else:
                # Warm up on a side stream so lazily created optimizer/compiler state is not
                # tied to the capture stream.
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    out = self.fn(*self.inputs)
                torch.cuda.current_stream().wait_stream(side)
                side.synchronize()
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"startup: {self.fn.__name__} warm-up {self.calls}/{self.warmup} took {time.monotonic() - begin:.2f} s", flush=True)
            return self._export(out) if external else out
        if self.capture_after is not None and self.capture_after.graph is None:
            return self._export(self.fn(*self.inputs))
        if self.graph is None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            begin = time.monotonic()
            self.graph = torch.cuda.CUDAGraph()
            options = dict(pool=self.pool.token, stream=self.pool.stream) if external else {}
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local", **options), self._capture_warp():
                if external:
                    out = self.fn(*self.inputs)
                    self._export(out)
                    del out
                else:
                    self.out = self.fn(*self.inputs)
            print(f"cuda_graph: captured {self.fn.__name__} in {time.monotonic() - begin:.2f} s", flush=True)
        self.graph.replay()
        self.replays += 1
        return tree_map(lambda x: x.detach().clone() if self.simulation is not None else x.detach(), self.out)
