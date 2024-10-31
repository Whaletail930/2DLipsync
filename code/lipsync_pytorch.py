import torch

from mfcc_extractor_lib import process_live


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load('lipsync_model_entire.pth', map_location=device)

    model.to(device)

    process_live(model, device)
