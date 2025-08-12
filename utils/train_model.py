from tqdm.auto import tqdm
import torch


def train_step(device, model, dataloader, loss_fn, optimizer):
    model.train()
    train_loss, train_acc = 0, 0

    for batch_num, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_labels = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_labels == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def test_step(device, model, dataloader, loss_fn):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch_num, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_labels = torch.argmax(y_pred, dim=1)
            test_acc += (y_pred_labels == y).sum().item() / len(y_pred)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


def train(device, model, train_loader, val_loader, optimizer, loss_fn, epochs):
    results = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            device, model, dataloader=train_loader,
            loss_fn=loss_fn, optimizer=optimizer
        )
        val_loss, val_acc = test_step(
            device, model, dataloader=val_loader, loss_fn=loss_fn
        )

        print(
            f"epoch: {epoch + 1} | train_loss {train_loss:.4f}",
            f"| train_acc: {train_acc:.4f}",
            f"| val_loss: {val_loss:.4f} | val_acc {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    return results
