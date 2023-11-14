import argparse
from Player import Player
from gametree.State import BLACK, WHITE
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Best Tablut player of the world (when it doesn't lose)")
    parser.add_argument("-i", "--ip", type=str, default="localhost", help="IP address of the hosting server")
    parser.add_argument("-p", "--port", type=str, default=None, help="Port of the hosting server")
    parser.add_argument("-c", "--color", type=str.lower, required=True, choices=["white", "black"], help="Color of the player")
    parser.add_argument("-t", "--timeout", type=int, default=60, help="Time available to make a decision")
    parser.add_argument("--tol", type=int, default=1, help="Tolerance on the timeout")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug logs")
    args = parser.parse_args()

    my_color = WHITE if args.color == "white" else BLACK
    player_name = 'TheCatIsOnTheTablut'

    player = Player(
        my_color = my_color,
        timeout = args.timeout,
        timeout_tol = args.tol,
        server_ip = args.ip,
        server_port = args.port,
        debug = args.debug,
    )

    player.play()