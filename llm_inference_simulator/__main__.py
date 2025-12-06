"""Allow running as: python -m llm_inference_simulator.cli"""
from .cli import main
import sys

if __name__ == '__main__':
    sys.exit(main())
