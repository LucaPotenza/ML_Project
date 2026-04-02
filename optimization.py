# Optimization
def build_cosine_warmup_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio=0.01):
    """
    LR schedule:
      - linear warmup from 0 -> 1 over warmup_steps
      - cosine decay from 1 -> min_lr_ratio over the remaining steps
    Returns: torch.optim.lr_scheduler.LambdaLR
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)  # 0..1
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1..0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine  # 1..min_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)