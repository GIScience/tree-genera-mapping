import torchinfo
import torch
def summary(model,
            batch_size=4,
            in_channels=5,
            img_height=640,
            img_width=640,
            device='cpu',
            verbose=True):
    """
    Prints a summary of the model using torchinfo.

    Args:
        model (nn.Module): PyTorch model
        batch_size (int): Batch size
        in_channels (int): Number of input channels
        img_height (int): Input image height
        img_width (int): Input image width
        device (str): 'cpu' or 'cuda'
        verbose (bool): Whether to print the summary
    """
    if isinstance(model, tuple):  # Unwrap if model is inside a tuple
        model = model[0]

    input_size = (batch_size, in_channels, img_height, img_width)

    return torchinfo.summary(
        model,
        input_size=input_size,
        device=device,
        row_settings=["var_names"],
        verbose=verbose
    )