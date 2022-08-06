from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from torch.autograd.function import once_differentiable

import hivemind
from load_balancer import LoadBalancer, NoModulesFound
from hivemind.moe.client.expert import DUMMY, expert_forward
from hivemind.compression import serialize_torch_tensor
from hivemind.utils import get_logger, nested_compare, nested_flatten, nested_pack, use_hivemind_log_handler


logger = get_logger(__name__)


MAX_PROMPT_LENGTH = 512
MAX_NODES = 99999


class DiffusionClient:
    def __init__(
        self,
        *,
        initial_peers: List[str],
        dht_prefix: str = "diffusion",
        **kwargs
    ):
        dht = hivemind.DHT(initial_peers, client_mode=True, start=True, **kwargs)
        self.expert = BalancedRemoteExpert(dht=dht, uid_prefix=dht_prefix + ".")

    def draw(self, prompts: List[str]) -> np.ndarray:
        encoded_prompts = []
        for prompt in prompts:
            tensor = torch.tensor(list(prompt.encode()), dtype=torch.int64)
            tensor = F.pad(tensor, (0, MAX_PROMPT_LENGTH - len(tensor)))
            encoded_prompts.append(tensor)
        encoded_prompts = torch.stack(encoded_prompts)

        output_images, = self.expert(encoded_prompts)
        return output_images.numpy()


class BalancedRemoteExpert(nn.Module):
    """
    A torch module that dynamically assigns weights to one RemoteExpert from a pool, proportionally to their throughput.
    ToDo docstring, similar to hivemind.RemoteExpert
    """

    def __init__(
        self,
        *,
        dht: hivemind.DHT,
        uid_prefix: str,
        grid_size: Tuple[int, ...] = (1, MAX_NODES),
        forward_timeout: Optional[float] = None,
        backward_timeout: Optional[float] = None,
        update_period: float = 30.0,
        backward_task_size_multiplier: float = 2.5,
        **kwargs,
    ):
        super().__init__()
        if uid_prefix.endswith(".0."):
            logger.warning(f"BalancedRemoteExperts will look for experts under prefix {self.uid_prefix}0.")
        assert len(grid_size) == 2 and grid_size[0] == 1, "only 1xN grids are supported"
        self.dht, self.uid_prefix, self.grid_size = dht, uid_prefix, grid_size
        self.forward_timeout, self.backward_timeout = forward_timeout, backward_timeout
        self.backward_task_size_multiplier = backward_task_size_multiplier
        self.expert_balancer = LoadBalancer(dht, key=f"{self.uid_prefix}0.", update_period=update_period, **kwargs)
        self._expert_info = None  # expert['info'] from one of experts in the grid

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor):
        """
        Call one of the RemoteExperts for the specified inputs and return output. Compatible with pytorch.autograd.

        :param args: input tensors that will be passed to each expert after input, batch-first
        :param kwargs: extra keyword tensors that will be passed to each expert, batch-first
        :returns: averaged predictions of all experts that delivered result on time, nested structure of batch-first
        """
        assert len(kwargs) == len(self.info["keyword_names"]), f"Keyword args should be {self.info['keyword_names']}"
        kwargs = {key: kwargs[key] for key in self.info["keyword_names"]}

        if self._expert_info is None:
            raise NotImplementedError()
        # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors

        forward_inputs = (args, kwargs)

        if not nested_compare(forward_inputs, self.info["forward_schema"]):
            raise TypeError(f"Inputs do not match expert input schema. Did you pass the right number of parameters?")

        flat_inputs = list(nested_flatten(forward_inputs))
        forward_task_size = flat_inputs[0].shape[0]

        # Note: we send DUMMY to prevent torch from excluding expert from backward if no other inputs require grad
        flat_outputs = _BalancedRemoteModuleCall.apply(DUMMY,
                                                       self.expert_balancer,
                                                       self.info,
                                                       self.forward_timeout,
                                                       self.backward_timeout,
                                                       forward_task_size,
                                                       forward_task_size * self.backward_task_size_multiplier,
                                                       *flat_inputs)

        return nested_pack(flat_outputs, structure=self.info["outputs_schema"])

    @property
    def info(self):
        while self._expert_info is None:
            try:
                with self.expert_balancer.use_another_expert(1) as chosen_expert:
                    self._expert_info = chosen_expert.info
            except NoModulesFound:
                raise
            except Exception:
                logger.exception(f"Tried to get expert info from {chosen_expert} but caught:")
        return self._expert_info


class _BalancedRemoteModuleCall(torch.autograd.Function):
    """Internal autograd-friendly call of a remote module. For applications, use BalancedRemoteExpert instead."""

    @staticmethod
    def forward(
            ctx,
            dummy: torch.Tensor,
            expert_balancer: LoadBalancer,
            info: Dict[str, Any],
            forward_timeout: float,
            backward_timeout: float,
            forward_task_size: float,
            backward_task_size: float,
            *inputs: torch.Tensor,
            ) -> Tuple[torch.Tensor, ...]:
        # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
        # detach to avoid pickling the computation graph
        ctx.expert_balancer, ctx.info = expert_balancer, info
        ctx.forward_timeout, ctx.backward_timeout = forward_timeout, backward_timeout
        ctx.forward_task_size, ctx.backward_task_size = forward_task_size, backward_task_size
        inputs = tuple(tensor.cpu().detach() for tensor in inputs)
        ctx.save_for_backward(*inputs)

        serialized_tensors = [
            serialize_torch_tensor(inp, proto.compression)
            for inp, proto in zip(inputs, nested_flatten(info["forward_schema"]))
        ]
        while True:
            try:
                with expert_balancer.use_another_expert(forward_task_size) as chosen_expert:
                    logger.info(f"Query served by: {chosen_expert}")
                    deserialized_outputs = RemoteExpertWorker.run_coroutine(expert_forward(
                        chosen_expert.uid, inputs, serialized_tensors, chosen_expert.stub))
                break
            except NoModulesFound:
                raise
            except Exception:
                logger.exception(f"Tried to call forward for expert {chosen_expert} but caught:")

        return tuple(deserialized_outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs) -> Tuple[Optional[torch.Tensor], ...]:
        raise NotImplementedError("Backward is not yet implemented in this example")
        # grad_outputs_cpu = tuple(tensor.cpu() for tensor in grad_outputs)
        # inputs_and_grad_outputs = tuple(nested_flatten((ctx.saved_tensors, grad_outputs_cpu)))
        # backward_schema = tuple(nested_flatten((ctx.info["forward_schema"], ctx.info["outputs_schema"])))
        # serialized_tensors = [
        #     serialize_torch_tensor(tensor, proto.compression)
        #     for tensor, proto in zip(inputs_and_grad_outputs, backward_schema)
        # ]
        # while True:
        #     try:
        #         with ctx.expert_balancer.use_another_expert(ctx.backward_task_size) as chosen_expert:
        #             backward_request = runtime_pb2.ExpertRequest(uid=chosen_expert.uid, tensors=serialized_tensors)
        #             grad_inputs = chosen_expert.stub.forward(backward_request, timeout=ctx.backward_timeout)
        #         break
        #     except NoModulesFound:
        #         raise
        #     except Exception:
        #         logger.exception(f"Tried to call backward for expert {chosen_expert} but caught:")
        # deserialized_grad_inputs = [deserialize_torch_tensor(tensor) for tensor in grad_inputs.tensors]
        # return (DUMMY, None, None, None, None, None, None, *deserialized_grad_inputs)
