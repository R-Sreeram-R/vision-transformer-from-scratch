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

def train(model, train_loader, test_loader, optimizer, loss_fn):
    ...
