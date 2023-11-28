import argparse
from Player import Player
import json
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Best Tablut player of the world (when it doesn't lose)")
    parser.add_argument("-i", "--ip", type=str, default="localhost", help="IP address of the hosting server")
    parser.add_argument("-p", "--port", type=str, default=None, help="Port of the hosting server")
    parser.add_argument("-c", "--color", type=str.lower, required=True, choices=["white", "black"], help="Color of the player")
    parser.add_argument("-t", "--timeout", type=int, default=60, help="Time available to make a decision")
    parser.add_argument("-w", "--weights", type=str, default="./weights.json", help="Weights to load")
    parser.add_argument("--tt-size", type=int, default=1e6, help="Number of entries in the transposition table")
    parser.add_argument("--tol", type=int, default=2, help="Tolerance on the timeout")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug logs")
    args = parser.parse_args()

    player_name = 'TheCatIsOnTheTablut'
    with open(args.weights, "r") as f:
        weights = json.load(f)

    player = Player(
        my_color = args.color,
        timeout = args.timeout,
        timeout_tol = args.tol,
        weights = weights[args.color],
        tt_size = args.tt_size,
        server_ip = args.ip,
        server_port = args.port,
        debug = args.debug,
    )

    player.play()