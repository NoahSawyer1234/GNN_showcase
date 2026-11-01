"""Node-level tasks with a GCN.

  python train_node.py --task classification   # Cora node classification
  python train_node.py --task regression       # demo regression on Cora

For regression the demo target is log(1 + degree) (standardised), predicted
from node features + graph structure. Swap in your own `data.y` for real use.
"""
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree

from models import NodeGCN


def load_data(task, root):
    dataset = Planetoid(root, "Cora", transform=T.NormalizeFeatures())
    data = dataset[0]
    if task == "regression":
        deg = degree(data.edge_index[0], data.num_nodes)
        y = torch.log1p(deg)
        data.y = ((y - y[data.train_mask].mean()) / y[data.train_mask].std()).unsqueeze(-1)
        out_channels = 1
    else:
        out_channels = dataset.num_classes
    return data, dataset.num_features, out_channels


def compute_loss(out, y, task):
    return F.cross_entropy(out, y) if task == "classification" else F.mse_loss(out, y)


def train(model, data, optimizer, task):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = compute_loss(out[data.train_mask], data.y[data.train_mask], task)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, task):
    model.eval()
    out = model(data.x, data.edge_index)
    res = {}
    for name, mask in [("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)]:
        if task == "classification":
            res[name] = (out[mask].argmax(-1) == data.y[mask]).float().mean().item()  # accuracy
        else:
            res[name] = F.mse_loss(out[mask], data.y[mask]).item()  # MSE
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--root", default="data")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, in_ch, out_ch = load_data(args.task, args.root)
    data = data.to(device)
    model = NodeGCN(in_ch, args.hidden, out_ch, args.layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    higher_better = args.task == "classification"
    metric = "acc" if higher_better else "mse"
    best_val, best_test = None, None
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, opt, args.task)
        r = evaluate(model, data, args.task)
        if best_val is None or (r["val"] > best_val if higher_better else r["val"] < best_val):
            best_val, best_test = r["val"], r["test"]
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} loss {loss:.4f} | {metric} train {r['train']:.4f} "
                  f"val {r['val']:.4f} test {r['test']:.4f}")
    print(f"best val {metric} {best_val:.4f} -> test {metric} {best_test:.4f}")


if __name__ == "__main__":
    main()
