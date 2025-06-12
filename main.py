import sys
import torch

def main():
    print("Hello from ucct!")
    print(sys.version)
    print(sys.executable)
    print(torch.__version__)
    print(torch.version.cuda)


if __name__ == "__main__":
    main()
