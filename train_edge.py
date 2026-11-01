"""Edge-level tasks with a GCN.

  python train_edge.py --task link_prediction     # predict held-out edges (AUC / AP)
  python train_edge.py --task classification      # classify edges

Link prediction follows the standard PyG recipe: split edges with
RandomLinkSplit, message-pass only over training edges, sample fresh negative
edges every epoch, and score pairs by dot product.

Edge classification demo on Cora: label an edge 1 if its endpoints share a
class, else 0. The model only sees node features and the *training* edges (never
node labels), so this is a fair homophily-prediction demo. Replace the labels
with your own `edge_y` for real use.
"""
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling, to_undirected

from models import EdgeGCN


# ----------------------------- link prediction -----------------------------
def run_link_prediction(args, device):
    dataset = Planetoid(args.root, "Cora", transform=T.NormalizeFeatures())
    split = T.RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True,
                              add_negative_train_samples=False, neg_sampling_ratio=1.0)
    train_data, val_data, test_data = [d.to(device) for d in split(dataset[0])]

    model = EdgeGCN(dataset.num_features, args.hidden, args.embed, args.layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train():
        model.train()
        opt.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        pos = train_data.edge_label_index
        neg = negative_sampling(train_data.edge_index, num_nodes=train_data.num_nodes,
                                num_neg_samples=pos.size(1))
        idx = torch.cat([pos, neg], dim=1)
        label = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]).to(device)
        loss = F.binary_cross_entropy_with_logits(model.decode_link(z, idx), label)
        loss.backward()
        opt.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(data):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        score = model.decode_link(z, data.edge_label_index).sigmoid().cpu().numpy()
        y = data.edge_label.cpu().numpy()
        return roc_auc_score(y, score), average_precision_score(y, score)

    best_val, best_test = 0, (0, 0)
    for epoch in range(1, args.epochs + 1):
        loss = train()
        val_auc, val_ap = evaluate(val_data)
        if val_auc > best_val:
            best_val, best_test = val_auc, evaluate(test_data)
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} loss {loss:.4f} val AUC {val_auc:.4f} AP {val_ap:.4f}")
    print(f"best val AUC {best_val:.4f} -> test AUC {best_test[0]:.4f}, AP {best_test[1]:.4f}")


# --------------------------- edge classification ---------------------------
def run_edge_classification(args, device):
    dataset = Planetoid(args.root, "Cora", transform=T.NormalizeFeatures())
    data = dataset[0]

    # One entry per undirected edge, with a demo label (same class or not).
    row, col = data.edge_index
    keep = row < col
    pairs = data.edge_index[:, keep]
    edge_y = (data.y[pairs[0]] == data.y[pairs[1]]).long()

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(pairs.size(1), generator=g)
    n_train, n_val = int(0.7 * len(perm)), int(0.15 * len(perm))
    parts = {"train": perm[:n_train], "val": perm[n_train:n_train + n_val], "test": perm[n_train + n_val:]}

    # Message passing uses training edges only, so val/test edges are unseen.
    mp_edge_index = to_undirected(pairs[:, parts["train"]]).to(device)
    x = data.x.to(device)
    pairs, edge_y = pairs.to(device), edge_y.to(device)
    parts = {k: v.to(device) for k, v in parts.items()}
    num_classes = int(edge_y.max()) + 1

    model = EdgeGCN(dataset.num_features, args.hidden, args.embed, args.layers,
                    args.dropout, num_edge_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    def train():
        model.train()
        opt.zero_grad()
        z = model.encode(x, mp_edge_index)
        idx = parts["train"]
        loss = F.cross_entropy(model.classify_edges(z, pairs[:, idx]), edge_y[idx])
        loss.backward()
        opt.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        z = model.encode(x, mp_edge_index)
        idx = parts[split]
        pred = model.classify_edges(z, pairs[:, idx]).argmax(-1).cpu().numpy()
        y = edge_y[idx].cpu().numpy()
        return (pred == y).mean(), f1_score(y, pred, average="macro")

    best_val, best_test = -1, None
    for epoch in range(1, args.epochs + 1):
        loss = train()
        val_acc, val_f1 = evaluate("val")
        if val_acc > best_val:
            best_val, best_test = val_acc, evaluate("test")
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} loss {loss:.4f} val acc {val_acc:.4f} macro-F1 {val_f1:.4f}")
    print(f"best val acc {best_val:.4f} -> test acc {best_test[0]:.4f}, macro-F1 {best_test[1]:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["link_prediction", "classification"], default="link_prediction")
    p.add_argument("--root", default="data")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--embed", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.task == "link_prediction":
        run_link_prediction(args, device)
    else:
        run_edge_classification(args, device)


if __name__ == "__main__":
    main()
