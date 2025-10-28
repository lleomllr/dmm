import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from model.model import DMM
from model.elbo import loss as dmm_loss 
from data.polyphonic import PolyDataset, NOTTINGHAM  


def collate_fn(batch):
    """
    Prend une liste d'éléments [(idx, seq, length), ...]
    et retourne des tenseurs padés + longueurs.
    """
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    idxs, seqs, lengths = zip(*batch)
    seqs = pad_sequence(seqs, batch_first=True)
    lengths = torch.stack(lengths)
    return seqs, lengths


def train_one_epoch(model, dataloader, optimizer, device, anneal_rate):
    model.train()
    total_loss, total_kl, total_nll = 0, 0, 0

    for x, lengths in tqdm(dataloader, desc="Training", leave=False):
        x = x.to(device)
        x_rev = torch.flip(x, dims=[1])  
        lengths = lengths.to(device)

        
        x_reco, z_q_seq, z_p_seq, mu_q, logvar_q, mu_p, logvar_p = model(x, x_rev, lengths)

        out = dmm_loss(
            x_hat=x_reco,
            x=x,
            mu_q=mu_q,
            mu_p=mu_p,
            logvar_q=logvar_q,
            logvar_p=logvar_p,
            annealing_factor=anneal_rate,
            logits=True
        )

        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += out["loss"].item()
        total_kl += out["kl_mean"].item()
        total_nll += out["nll_mean"].item()

    return total_loss / len(dataloader), total_nll / len(dataloader), total_kl / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_kl, total_nll = 0, 0, 0
    for x, lengths in tqdm(dataloader, desc="Validation", leave=False):
        x = x.to(device)
        x_rev = torch.flip(x, dims=[1])
        lengths = lengths.to(device)

        x_reco, z_q_seq, z_p_seq, mu_q, logvar_q, mu_p, logvar_p = model(x, x_rev, lengths)
        out = dmm_loss(
            x_hat=x_reco,
            x=x,
            mu_q=mu_q,
            mu_p=mu_p,
            logvar_q=logvar_q,
            logvar_p=logvar_p,
            logits=True
        )

        total_loss += out["loss"].item()
        total_kl += out["kl_mean"].item()
        total_nll += out["nll_mean"].item()

    return total_loss / len(dataloader), total_nll / len(dataloader), total_kl / len(dataloader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training DMM on {args.dataset} — using device {device}")

    # --- Load dataset
    dataset = NOTTINGHAM if args.dataset == "nottingham" else JSB_CHORALES
    train_data = PolyDataset(dataset, split="train")
    valid_data = PolyDataset(dataset, split="valid")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = DMM(
        x_dim=88,
        z=args.z_dim,
        emission_dim=args.emission_dim,
        transition_dim=args.transition_dim,
        rnn_dim=args.rnn_dim,
        rnn_type="gru",
        rnn_layers=1,
        gated=True,
        train_init=True,
        comb_hidden=args.comb_hidden,
        use_stlr=False
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        anneal = min(1.0, epoch / args.anneal_epochs)
        train_loss, train_nll, train_kl = train_one_epoch(model, train_loader, optimizer, device, anneal)
        val_loss, val_nll, val_kl = evaluate(model, valid_loader, device)

        print(f"[Epoch {epoch:02d}] "
              f"Train Loss={train_loss:.3f} (NLL={train_nll:.3f}, KL={train_kl:.3f}) | "
              f"Val Loss={val_loss:.3f} (NLL={val_nll:.3f}, KL={val_kl:.3f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"dmm_{args.dataset}_best.pt")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Deep Markov Model (DMM) on polyphonic data")
    parser.add_argument("--dataset", type=str, default="nottingham", help="Dataset: nottingham | jsb_chorales")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--z-dim", type=int, default=50)
    parser.add_argument("--emission-dim", type=int, default=200)
    parser.add_argument("--transition-dim", type=int, default=200)
    parser.add_argument("--rnn-dim", type=int, default=400)
    parser.add_argument("--comb-hidden", type=int, default=200)
    parser.add_argument("--anneal-epochs", type=int, default=20)
    args = parser.parse_args()

    main(args)
