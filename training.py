# Epoch training
def train_epoch(model, train_loader, optimizer, prm, epoch, scheduler=None):
    """Esegue una singola epoca di training"""

    model.train()

    epoch_metrics = {
        'cost': 0, 'cost_pos': 0, 'cost_neg': 0,
        'dpos': 0, 'dneg': 0, 'dpos_sp': 0, 'dneg_sp': 0,
        'dpos_vp': 0, 'dneg_vp': 0, 'posdiff': 0, 'negdiff': 0,
        'numpos': 0, 'numneg': 0
    }

    for batch_idx, batch in enumerate(train_loader):
        # Sposta dati su device
        sketch = batch['sketch'].to(prm.device)
        view = batch['view'].to(prm.device)
        sketch_peer = batch['sketch_peer'].to(prm.device)
        view_peer = batch['view_peer'].to(prm.device)
        sketch_labels = batch['sketch_label'].to(prm.device)
        view_labels = batch['view_label'].to(prm.device)

        # Forward pass
        if prm.model_type == 'dual':
            sketch_emb = model.encode_sketch(sketch)
            view_emb = model.encode_view(view)
            sketch_peer_emb = model.encode_sketch(sketch_peer)
            view_peer_emb = model.encode_view(view_peer)
        else:  # shared
            sketch_emb = model(sketch)
            view_emb = model(view)
            sketch_peer_emb = model(sketch_peer)
            view_peer_emb = model(view_peer)

        # Sampling stocastico (maschere)
        batch_size = sketch.size(0)
        mask_pos = torch.bernoulli(torch.full((batch_size, batch_size), prm.ppos, device=prm.device)).float()
        mask_neg = torch.bernoulli(torch.full((batch_size, batch_size), prm.pneg, device=prm.device)).float()


        # Calcola loss
        loss, metrics = contrastive_loss(
            sketch_emb, view_emb, sketch_peer_emb, view_peer_emb,
            sketch_labels, view_labels,
            Cpos=prm.Cpos, Cneg=prm.Cneg, alpha=prm.alpha,
            sw=prm.sw, vw=prm.vw,
            mask_pos=mask_pos, mask_neg=mask_neg
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step LR scheduler per-iteration (IMPORTANT: after optimizer.step())
        if scheduler is not None:
            scheduler.step()


        # Accumula metriche
        for key in epoch_metrics:
            epoch_metrics[key] += metrics[key]

        # Logging ogni 20 batch
        if batch_idx % 20 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            iter_num = epoch * len(train_loader) + batch_idx
            print(f'  {iter_num:8d} | {metrics["cost"]:6.6f} | lr {current_lr:.2e} |'
                  f'c  {metrics["dpos"]/max(metrics["numpos"],1):2.6f} | {metrics["dneg"]/max(metrics["numneg"],1):2.6f} |'
                  f's  {metrics["dpos_sp"]/max(metrics["numpos"],1):2.6f} | {metrics["dneg_sp"]/max(metrics["numneg"],1):2.6f} |'
                  f'v  {metrics["dpos_vp"]/max(metrics["numpos"],1):2.6f} | {metrics["dneg_vp"]/max(metrics["numneg"],1):2.6f} |'
                  f' {metrics["posdiff"]:2.6f} |p {metrics["posdiff"]:2.6f} |n {metrics["negdiff"]:2.6f} |'
                  f' {int(metrics["numpos"]):4d} | {int(metrics["numneg"]):4d}')

    # Media metriche
    n_batches = len(train_loader)
    for key in epoch_metrics:
        epoch_metrics[key] /= n_batches

    return epoch_metrics

# Train model
def train_model(prm, sketches, views, sketch_label, view_label, triples, labels):
    """Funzione principale di training"""

    # Sanity checks (fail fast)
    assert sketches.ndim == 2 and views.ndim == 2, "Expected flat vectors in .mat datax"
    assert triples.shape[1] == 4, "triples must have 4 indices: [sketch, view, sketch_peer, view_peer]"
    assert labels.shape[1] >= 3, "labels must include at least [pos/neg, sketch_class, view_class]"

    print(f'Loaded {sketches.shape[0]} sketches, {views.shape[0]} views')
    print(f'Loaded {len(triples)} training pairs')

    # Crea dataset e dataloader
    dataset = SketchViewDataset(sketches, views, triples, labels)
    train_loader = DataLoader(
        dataset,
        batch_size=prm.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1), # pytorch gave me some errors so I fixed it
        pin_memory=torch.cuda.is_available()  # True solo se c'è una GPU
    )

    total_steps = prm.n_epochs * len(train_loader)
    warmup_steps = int(prm.warmup_ratio * total_steps)
    print(f"[INFO] LR schedule: total_steps={total_steps}, warmup_steps={warmup_steps}, min_lr_ratio={prm.min_lr_ratio}")

    # Crea modello
    if prm.model_type == 'dual':
        model = DualSketchCNN(
            code_len=prm.code_len,
            use_dropout=prm.use_dropout,
            dropout_p=prm.dropout_p
        ).to(prm.device)
    else:
        model = SketchCNN(
            code_len=prm.code_len,
            use_dropout=prm.use_dropout,
            dropout_p=prm.dropout_p
        ).to(prm.device)

    # Optimizer
    optimizer = optim.AdamW(
    model.parameters(),
    lr=prm.learning_rate,
    weight_decay=prm.weight_decay,
    betas=(0.9, 0.999)
    )

    scheduler = build_cosine_warmup_scheduler(
    optimizer=optimizer,
    total_steps=total_steps,
    warmup_steps=warmup_steps,
    min_lr_ratio=prm.min_lr_ratio
    )

    print('Starting training...\n')

    # Header tabella monitoring
    print('     #iter     |   cost    |     dpos   |   dneg   |   dpos_sp  | dneg_sp  |'
          '   dpos_vp  | dneg_vp  |   diff   |  posdiff  |  negdiff  | #pos | #neg')

    # Training loop
    best_loss = float('inf')
    best_epoch_saved = None # Initialize a variable to store the best epoch
    start_time = time.time()

    for epoch in range(1, prm.n_epochs + 1):
        # Train
        epoch_metrics = train_epoch(model, train_loader, optimizer, prm, epoch, scheduler=scheduler)

        # Print epoch summary
        print(f'\nEpoch {epoch}/{prm.n_epochs} completed. Avg Loss: {epoch_metrics["cost"]:.6f}')

        # Salva checkpoint
        if epoch_metrics['cost'] < best_loss:
            best_loss = epoch_metrics['cost']
            best_epoch_saved = epoch # Update best epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'parameters': vars(prm)
            }

            save_path = os.path.join(prm.model_dir, f'{prm.exp_name}-{prm.exp_suffix}-epoch{epoch}.pth')
            torch.save(checkpoint, save_path)
            print(f'Saved best model to {save_path}')

        elapsed = (time.time() - start_time) / 60
        print(f'Training time elapsed: {elapsed:.1f} min\n')

    print(f'Training complete! Best loss: {best_loss:.6f} at epoch {best_epoch_saved}')
    return best_epoch_saved # Return the best epoch

# EXPERIMENT: run training
def run_case(prm: Parameters, run_mode: str, all_data: dict):
    """
    Run a single case in train or test mode.
    Returns best_epoch_from_train if run_mode is "train".
    """
    ensure_dir(prm.model_dir)
    ensure_dir(prm.feats_dir)

    print("\n" + "=" * 80)
    print(f"CASE: {prm.exp_name} / {prm.exp_suffix} | mode={run_mode} | device={prm.device}")
    print("=" * 80)
    pprint(vars(prm))

    best_epoch_from_train = None

    if run_mode == "train":
        best_epoch_from_train = train_model(
            prm,
            sketches=all_data['train_sketches'],
            views=all_data['views'],
            sketch_label=all_data['train_sketch_labels'],
            view_label=all_data['view_labels'],
            triples=all_data['train_triples'],
            labels=all_data['train_labels']
        )
    # The 'test' logic will now be handled directly in main or by an explicit call outside.
    else:
        pass

    return best_epoch_from_train # Return best epoch from training, will be None if not training