import torch
from tqdm.auto import tqdm

def train_step(model, dataloader, loss_fn, optimizer, device, epoch):

    model.train()

    train_loss, train_acc = 0, 0

    # Mini-batch progress bar
    batch_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch} [Train]",
        leave=False
    )

    for batch, (X, y) in batch_bar:

        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        batch_acc = (y_pred_class == y).sum().item() / len(y_pred)

        train_acc += batch_acc

        # Update live metrics in bar
        batch_bar.set_postfix({
            "batch": f"{batch+1}/{len(dataloader)}",
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_acc:.4f}"
        })

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, device, epoch):

    model.eval()

    test_loss, test_acc = 0, 0

    batch_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch} [Test]",
        leave=False
    )

    with torch.inference_mode():

        for batch, (X, y) in batch_bar:

            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            batch_acc = (test_pred_labels == y).sum().item() / len(test_pred_labels)

            test_acc += batch_acc

            batch_bar.set_postfix({
                "batch": f"{batch+1}/{len(dataloader)}",
                "loss": f"{loss.item():.4f}",
                "acc": f"{batch_acc:.4f}"
            })

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


def train(model, train_loader, test_loader, optimizer, loss_fn, epochs, device):

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Epoch-level progress bar
    epoch_bar = tqdm(range(epochs), desc="Training Progress")

    for epoch in epoch_bar:

        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch+1
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch+1
        )

        epoch_bar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.4f}",
            "test_loss": f"{test_loss:.4f}",
            "test_acc": f"{test_acc:.4f}"
        })

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
