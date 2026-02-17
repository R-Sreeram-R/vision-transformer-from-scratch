import torch
from tqdm.auto import tqdm

def train_step(model, dataloader, loss_fn, optimizer, device, epoch, epoch_bar=None):

    model.train()
    train_loss, train_acc = 0, 0

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

        batch_bar.set_postfix({
            "batch": f"{batch+1}/{len(dataloader)}",
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_acc:.4f}"
        })

        # ✅ Update the *epoch-level* bar every minibatch
        if epoch_bar is not None:
            epoch_bar.update(1)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, device, epoch, epoch_bar=None):

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

            # ✅ Also update outer bar during evaluation (optional but feels alive)
            if epoch_bar is not None:
                epoch_bar.update(1)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model, train_loader, test_loader, optimizer, loss_fn, epochs, device):

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # ✅ Total steps = (train batches + test batches) per epoch * epochs
    total_steps = epochs * (len(train_loader) + len(test_loader))

    epoch_bar = tqdm(total=total_steps, desc="Training Progress", unit="batch")

    for epoch in range(epochs):

        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            epoch_bar=epoch_bar
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch + 1,
            epoch_bar=epoch_bar
        )

        # keep your print exactly the same
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

        epoch_bar.set_postfix({
            "epoch": f"{epoch+1}/{epochs}",
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.4f}",
            "test_loss": f"{test_loss:.4f}",
            "test_acc": f"{test_acc:.4f}",
        })

    epoch_bar.close()
    return results
