import matplotlib.pyplot as plt


def save_grid(x, path, num_classes=10, show_per_class=None):
    img = (x.clamp(-1, 1) + 1) / 2
    img = img.detach().cpu().numpy()
    n_per_class = img.shape[0] // num_classes
    img = img.reshape(num_classes, n_per_class, img.shape[-2], img.shape[-1])
    if show_per_class is not None:
        img = img[:, :show_per_class]
    rows = img.shape[1]

    fig, axes = plt.subplots(
        rows, num_classes,
        figsize=(num_classes, rows + 0.4),
        squeeze=False,
    )
    for c in range(num_classes):
        for r in range(rows):
            ax = axes[r, c]
            ax.imshow(img[c, r], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(str(c))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
