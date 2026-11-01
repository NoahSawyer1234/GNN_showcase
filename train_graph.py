"""Graph-level tasks with a GCN.

  python train_graph.py --task classification   # MUTAG (TUDataset), graph labels
  python train_graph.py --task regression       # ZINC subset, penalised logP

Each graph is encoded by GCN layers, pooled to a single vector (mean/sum/max),
and mapped to a class logit vector or a regression value by an MLP head.
"""
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ZINC, TUDataset
from torch_geometric.loader import DataLoader

from models import GraphGCN


def one_hot_atoms(data, num_types=28):
    """ZINC stores atom types as integer ids; turn them into one-hot features."""
    data.x = F.one_hot(data.x.view(-1), num_types).float()
    return data


def load_data(task, root, seed):
    if task == "classification":
        ds = TUDataset(root, "MUTAG").shuffle()
        n = len(ds)
        train, val, test = ds[: int(0.8 * n)], ds[int(0.8 * n): int(0.9 * n)], ds[int(0.9 * n):]
        in_ch, out_ch = ds.num_features, ds.num_classes
    else:
        train, val, test = (ZINC(f"{root}/ZINC", subset=True, split=s, transform=one_hot_atoms)
                            for s in ("train", "val", "test"))
        in_ch, out_ch = 28, 1
    return train, val, test, in_ch, out_ch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--root", default="data")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pooling", choices=["mean", "sum", "max"], default="mean")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds, in_ch, out_ch = load_data(args.task, args.root, args.seed)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, args.batch_size)
    test_loader = DataLoader(test_ds, args.batch_size)

    model = GraphGCN(in_ch, args.hidden, out_ch, args.layers, args.dropout, args.pooling).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    clf = args.task == "classification"

    def loss_fn(out, batch):
        return F.cross_entropy(out, batch.y) if clf else F.l1_loss(out.view(-1), batch.y.view(-1))

    def train():
        model.train()
        total = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch.x, batch.edge_index, batch.batch), batch)
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
        return total / len(train_loader.dataset)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        score = 0.0
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            if clf:
                score += (out.argmax(-1) == batch.y).sum().item()          # accuracy
            else:
                score += (out.view(-1) - batch.y.view(-1)).abs().sum().item()  # MAE
        return score / len(loader.dataset)

    metric = "acc" if clf else "MAE"
    best_val, best_test = None, None
    for epoch in range(1, args.epochs + 1):
        loss = train()
        val, test = evaluate(val_loader), evaluate(test_loader)
        if best_val is None or (val > best_val if clf else val < best_val):
            best_val, best_test = val, test
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} loss {loss:.4f} | val {metric} {val:.4f} test {metric} {test:.4f}")
    print(f"best val {metric} {best_val:.4f} -> test {metric} {best_test:.4f}")


if __name__ == "__main__":
    main()
