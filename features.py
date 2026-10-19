import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import canny, graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import sobel


FEATURE_NAMES = (
    [f"RGB_Mean_{c}" for c in "RGB"]
    + [f"RGB_Std_{c}" for c in "RGB"]
    + [f"HSV_Mean_{c}" for c in "HSV"]
    + [f"HSV_Std_{c}" for c in "HSV"]
    + ["ExG_Mean", "NGRDI_Mean"]
    + [f"GLCM_{p}_{d}" for p in
       ("Contrast", "Dissimilarity", "Homogen", "Energy", "Corr", "ASM")
       for d in ("d5", "d15", "d30")]
    + [f"LBP_bin_{i}" for i in range(26)]
    + ["Edge_Density_Sobel", "Edge_Std_Sobel", "Edge_Ratio_Canny"]
)


def extract_features(image):
    rgb = np.asarray(image.convert("RGB"))
    hsv = rgb2hsv(rgb)
    red, green, blue = (rgb[:, :, i].astype(float) for i in range(3))

    color = np.concatenate([
        rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1)),
        hsv.mean(axis=(0, 1)), hsv.std(axis=(0, 1)),
        [(2 * green - red - blue).mean(),
         ((green - red) / (green + red + 1e-6)).mean()],
    ])

    gray_float = rgb2gray(rgb)
    gray = (gray_float * 255).astype(np.uint8)
    glcm = graycomatrix(
        gray, distances=[5, 15, 30],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256, symmetric=True, normed=True,
    )
    glcm_values = np.concatenate([
        graycoprops(glcm, prop).mean(axis=1)
        for prop in ("contrast", "dissimilarity", "homogeneity",
                     "energy", "correlation", "ASM")
    ])

    lbp = local_binary_pattern(gray, 24, 3, method="uniform")
    lbp_values, _ = np.histogram(lbp.ravel(), bins=np.arange(27), density=True)
    texture = np.concatenate([glcm_values, lbp_values])

    edges = sobel(gray_float)
    structure = [edges.mean(), edges.std(), canny(gray_float, sigma=2.0).mean()]
    features = np.concatenate([color, texture, structure])

    if features.size != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError("Invalid extracted features")
    return features
