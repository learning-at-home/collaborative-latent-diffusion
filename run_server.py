from functools import partial
from pathlib import Path

import configargparse
import hivemind
import torch
from hivemind import ModuleBackend

from hivemind.moe import Server
from hivemind.moe.server.layers import add_custom_models_from_file, name_to_block, name_to_input
from hivemind.moe.server.server import _generate_uids
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.limits import increase_file_limit
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from hivemind.utils.tensor_descr import DUMMY_BATCH_SIZE, BatchTensorDescriptor

from client import MAX_NODES

use_hivemind_log_handler("in_root_logger")
logger = get_logger(__name__)


def main():
    # fmt:off
    parser = configargparse.ArgParser(default_config_files=["config.yml"])
    parser.add('-c', '--config', required=False, is_config_file=True, help='config file path')

    parser.add_argument('--dht_prefix', type=str, required=True)
    parser.add_argument('--module_cls', type=str, default='ffn', required=False,
                        help="name of a pytorch module that is being served, see your_code_here.py")
    parser.add_argument('--host_maddrs', type=str, nargs='+', default=['/ip4/0.0.0.0/tcp/0'], required=False,
                        help='Multiaddrs to listen for external connections from other p2p instances; default: all IPv4 and TCP: /ip4/0.0.0.0/tcp/0')
    parser.add_argument('--announce_maddrs', type=str, nargs='+', default=None, required=False,
                        help='Visible multiaddrs the host announces for external connections from other p2p instances')

    parser.add_argument('--num_handlers', type=int, default=8, required=False,
                        help='server will use this many processes to handle incoming requests')
    parser.add_argument('--min_batch_size', type=int, default=1,
                        help='Minimum required batch size for all expert operations')
    parser.add_argument('--max_batch_size', type=int, default=16,
                        help='The total number of examples in the same batch will not exceed this value')
    parser.add_argument('--device', type=str, default=None, required=False,
                        help='all experts will use this device in torch notation; default: cuda if available else cpu')

    parser.add_argument('--update_period', type=float, required=False, default=30,
                        help='Server will report experts to DHT once in this many seconds')
    parser.add_argument('--expiration', type=float, required=False, default=None,
                        help='DHT entries will expire after this many seconds')
    parser.add_argument('--initial_peers', type=str, nargs='*', required=False, default=[],
                        help='multiaddrs of one or more active DHT peers (if you want to join an existing DHT)')
    parser.add_argument('--increase_file_limit', action='store_true',
                        help='On *nix, this will increase the max number of processes '
                             'a server can spawn before hitting "Too many open files"; Use at your own risk.')
    parser.add_argument('--compression', type=str, default='NONE', required=False, help='Tensor compression for gRPC')
    parser.add_argument('--checkpoint_dir', type=Path, required=False, help='Directory to store expert checkpoints')
    parser.add_argument('--stats_report_interval', type=int, required=False,
                        help='Interval between two reports of batch processing performance statistics')

    parser.add_argument('--custom_module_path', type=str, required=False,
                        help='Path of a file with custom nn.modules, wrapped into special decorator')
    parser.add_argument('--identity_path', type=str, required=False, help='Path to identity file to be used in P2P')

    # fmt:on
    args = parser.parse_args()
    expert_pattern = f"{args.dht_prefix}.0.[0:{MAX_NODES}]"

    if args.increase_file_limit:
        increase_file_limit()

    compression = getattr(CompressionType, args.compression)
    if args.custom_module_path is not None:
        add_custom_models_from_file(args.custom_module_path)
    assert args.module_cls in name_to_block, f"module {args.module_cls} not found"

    dht = hivemind.DHT(initial_peers=args.initial_peers,
                       host_maddrs=args.host_maddrs,
                       announce_maddrs=args.announce_maddrs,
                       start=True)
    visible_maddrs_str = [str(a) for a in dht.get_visible_maddrs()]
    logger.info(f"Running DHT node on {visible_maddrs_str}, initial peers = {args.initial_peers}")

    num_modules = 1
    reserved_uids = []

    uids_to_generate = num_modules - len(reserved_uids)
    if uids_to_generate > 0:
        logger.info(f"Generating {uids_to_generate} expert uids from pattern {expert_pattern}")
        reserved_uids.extend(_generate_uids(uids_to_generate, expert_pattern, dht))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sample_input = name_to_input[args.module_cls](DUMMY_BATCH_SIZE)
    if isinstance(sample_input, tuple):
        args_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in sample_input)
    else:
        args_schema = (BatchTensorDescriptor.from_tensor(sample_input, compression),)

    # initialize pytorch module
    backends = {}
    for uid in reserved_uids:
        backends[uid] = ModuleBackend(
            name=uid,
            module=name_to_block[args.module_cls](),
            args_schema=args_schema,
            optimizer=None,
            scheduler=None,
            min_batch_size=args.min_batch_size,
            max_batch_size=args.max_batch_size,
        )

    server = Server(
        dht,
        backends,
        num_connection_handlers=args.num_handlers,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        stats_report_interval=args.stats_report_interval,
        update_period=args.update_period,
        expiration=args.expiration,
        start=True,
    )

    try:
        server.join()
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, shutting down")
    finally:
        server.shutdown()

if __name__ == "__main__":
    main()
