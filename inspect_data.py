from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def inspect_nifti(path: Path, is_mask: bool = False) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # shape: (z, y, x)

    print(f"\n--- {path.name} ---")
    print(f"Size (x, y, z): {img.GetSize()}")
    print(f"Array shape (z, y, x): {arr.shape}")
    print(f"Spacing (mm):   {tuple(round(v, 3) for v in img.GetSpacing())}")
    print(f"Origin:         {tuple(round(v, 2) for v in img.GetOrigin())}")
    print(f"Direction:      {tuple(round(v, 2) for v in img.GetDirection())}")
    print(f"Pixel type:     {img.GetPixelIDTypeAsString()}")

    if is_mask:
        unique_vals, counts = np.unique(arr, return_counts=True)
        print("Unique labels & voxel counts:")
        for val, cnt in zip(unique_vals, counts):
            print(f"  Label {int(val)}: {int(cnt)} voxels")
    else:
        print(f"Intensity min/max: {float(arr.min()):.1f} / {float(arr.max()):.1f}")
        print(f"Mean / Std:        {float(arr.mean()):.1f} / {float(arr.std()):.1f}")

    return arr


def show_ct_and_mask(ct_arr: np.ndarray, mask_arr: np.ndarray) -> None:
    z_mid = ct_arr.shape[0] // 2
    y_mid = ct_arr.shape[1] // 2
    x_mid = ct_arr.shape[2] // 2

    # Bone-friendly CT windowing
    vmin, vmax = -200.0, 800.0

    # Figure 1: Middle slices of CT in Axial, Coronal, and Sagittal views
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("common_40 CT Scan (Middle Slices)", fontsize=13)

    axes[0].imshow(ct_arr[z_mid, :, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Axial (z = {z_mid})")
    axes[0].axis("off")

    axes[1].imshow(ct_arr[:, y_mid, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Coronal (y = {y_mid})")
    axes[1].axis("off")

    axes[2].imshow(ct_arr[:, :, x_mid], cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Sagittal (x = {x_mid})")
    axes[2].axis("off")

    plt.tight_layout()

    # Figure 2: CT + Segmentation Mask Overlay across the three views
    masked_z = np.ma.masked_where(mask_arr[z_mid, :, :] == 0, mask_arr[z_mid, :, :])
    masked_y = np.ma.masked_where(mask_arr[:, y_mid, :] == 0, mask_arr[:, y_mid, :])
    masked_x = np.ma.masked_where(mask_arr[:, :, x_mid] == 0, mask_arr[:, :, x_mid])

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))
    fig2.suptitle("common_40 CT + Ground-Truth Mask Overlay", fontsize=13)

    axes2[0].imshow(ct_arr[z_mid, :, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes2[0].imshow(masked_z, cmap="tab10", alpha=0.5)
    axes2[0].set_title(f"Axial Overlay (z = {z_mid})")
    axes2[0].axis("off")

    axes2[1].imshow(ct_arr[:, y_mid, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes2[1].imshow(masked_y, cmap="tab10", alpha=0.5)
    axes2[1].set_title(f"Coronal Overlay (y = {y_mid})")
    axes2[1].axis("off")

    axes2[2].imshow(ct_arr[:, :, x_mid], cmap="gray", vmin=vmin, vmax=vmax)
    axes2[2].imshow(masked_x, cmap="tab10", alpha=0.5)
    axes2[2].set_title(f"Sagittal Overlay (x = {x_mid})")
    axes2[2].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    common_dir = project_dir / "COMMON_images_masks"

    ct_arr = inspect_nifti(common_dir / "common_40_image.nii", is_mask=False)
    mask_arr = inspect_nifti(common_dir / "common_40_mask.nii", is_mask=True)
    show_ct_and_mask(ct_arr, mask_arr)


if __name__ == "__main__":
    main()