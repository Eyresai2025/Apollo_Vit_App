import torch

def save_checkpoint(epoch, model, optimizer, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "loss": loss,
        },
        path,
    )
    print(f"Model saved at epoch {epoch + 1} to {path}")


def load_checkpoint(model, optimizer, path, strict=True):
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    opt_state = checkpoint.get("optimizer_state_dict", None)
    if optimizer is not None and opt_state is not None:
        optimizer.load_state_dict(opt_state)

    epoch = checkpoint.get("epoch", -1)
    loss = checkpoint.get("loss", None)

    if loss is not None:
        print(f"Loaded model from epoch {epoch + 1}, with loss: {loss:.6f}")
    else:
        print(f"Loaded model from epoch {epoch + 1}")

    return epoch
