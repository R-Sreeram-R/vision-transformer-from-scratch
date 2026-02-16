import torch 

from tqdm.auto import tqdm 

def train_step(model,dataloader,loss_fn,optimizer,device):

    model.train()

    train_loss, test_loss = 0,0 

    for batch, (X,y) in enumerate(dataloader):

        X,y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred,y)
        train_loss += loss 

        optimizer.zero_grad()

        loss.backward() 

        optimizer.step() 

        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model, dataloader, loss_fn,device):

  model.eval() 

  test_loss, test_acc = 0, 0


  with torch.inference_mode():
      
      for batch, (X, y) in enumerate(dataloader):
          
          X, y = X.to(device), y.to(device)

          test_pred_logits = model(X)

          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()

          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc


def train(model, train_loader, test_loader, optimizer, loss_fn, epochs, device):

    results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
    }

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(model=model, dataloader=train_loader, loss_fn=loss_fn, optimizer=optimizer, device = device)
        
        test_loss, test_acc = test_step(model=model,dataloader=test_loader,loss_fn=loss_fn,device=device)

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


