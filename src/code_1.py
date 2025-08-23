import io
import os
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add Gemini info helper
import google.generativeai as genai


# Set API key from environment variable (recommended)
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def get_gemini_info(transform_name: str) -> str:
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Explain the image transform: {transform_name} in 2-3 sentences.",
        )
        return response.text
    except Exception as e:
        return f"Gemini API error: {e}"

def get_gemini_explanation(transform_name: str):
    prompt = f"Explain {transform_name} in computer vision with real-world applications (like Tesla self-driving)."
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {str(e)}"

# Optional / extra goodies
try:
    from skimage.feature import graycomatrix, graycoprops, canny
    from skimage.color import rgb2gray
    from skimage.segmentation import slic, mark_boundaries
    from skimage.util import img_as_ubyte
    from skimage.measure import ransac, LineModelND, label, regionprops
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
except Exception:
    graycomatrix = None
    graycoprops = None
    canny = None
    rgb2gray = None
    slic = None
    mark_boundaries = None
    img_as_ubyte = None
    ransac = None
    LineModelND = None
    label = None
    regionprops = None
    PCA = None
    KMeans = None

###############################################################################
# Utility helpers
###############################################################################

def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


###############################################################################
# Transformation primitives
###############################################################################

def resize_image(img: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def change_colorspace(img: np.ndarray, space: str) -> np.ndarray:
    space = space.upper()
    if space == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if space == "HSV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if space == "LAB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if space == "YCrCb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    return img


def gaussian_blur(img: np.ndarray, k: int, sigma: float) -> np.ndarray:
    k = max(1, k)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def median_blur(img: np.ndarray, k: int) -> np.ndarray:
    k = max(1, k)
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(img, k)


def bilateral_filter(img: np.ndarray, d: int, sigmaColor: float, sigmaSpace: float) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)


def sobel_edges(img: np.ndarray) -> np.ndarray:
    gray = to_gray(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.uint8(255 * mag / (mag.max() + 1e-6))
    return mag


def canny_edges(img: np.ndarray, t1: int, t2: int) -> np.ndarray:
    gray = to_gray(img)
    return cv2.Canny(gray, t1, t2)


def threshold_binary(img: np.ndarray, method: str = "Otsu", thresh: int = 127) -> np.ndarray:
    gray = to_gray(img)
    if method == "Otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "Adaptive (mean)":
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    elif method == "Adaptive (gaussian)":
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    else:
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return bw


def morphology(img: np.ndarray, op: str, ksize: int = 3, iters: int = 1) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    gray = img if img.ndim == 2 else to_gray(img)
    if op == "erode":
        return cv2.erode(gray, kernel, iterations=iters)
    if op == "dilate":
        return cv2.dilate(gray, kernel, iterations=iters)
    if op == "open":
        return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iters)
    if op == "close":
        return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return gray


def kmeans_segmentation(img: np.ndarray, k: int = 3, attempts: int = 3) -> np.ndarray:
    data = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img.shape)
    return segmented


def slic_superpixels(img: np.ndarray, n_segments: int = 200, compactness: float = 10.0) -> np.ndarray:
    if slic is None or mark_boundaries is None:
        return img
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    seg = slic(rgb, n_segments=n_segments, compactness=compactness, start_label=1)
    vis = mark_boundaries(rgb, seg, mode='thick')
    vis_bgr = cv2.cvtColor((vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return vis_bgr


def harris_corners(img: np.ndarray, block: int = 2, ksize: int = 3, k: float = 0.04, thresh: float = 0.01) -> np.ndarray:
    gray = np.float32(to_gray(img))
    dst = cv2.cornerHarris(gray, block, ksize, k)
    dst = cv2.dilate(dst, None)
    out = img.copy()
    out[dst > thresh * dst.max()] = [0, 0, 255]
    return out


def shi_tomasi(img: np.ndarray, maxCorners: int = 200, qualityLevel: float = 0.01, minDistance: int = 10) -> np.ndarray:
    gray = to_gray(img)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)
    out = img.copy()
    if corners is not None:
        for c in np.int0(corners):
            x, y = c.ravel()
            cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
    return out


def sift_orb_features(img: np.ndarray, max_features: int = 400) -> np.ndarray:
    gray = to_gray(img)
    out = img.copy()
    kp = []
    try:
        sift = cv2.SIFT_create(nfeatures=max_features)
        kp = sift.detect(gray, None)
        out = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        label = f"SIFT keypoints: {len(kp)}"
    except Exception:
        # Fallback to ORB (free, widely available)
        orb = cv2.ORB_create(nfeatures=max_features)
        kp = orb.detect(gray, None)
        kp, des = orb.compute(gray, kp)
        out = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        label = f"ORB keypoints: {0 if kp is None else len(kp)}"
    return out


def pca_compress(img: np.ndarray, n_components: int = 32) -> Tuple[np.ndarray, float]:
    if PCA is None:
        return img, 1.0
    h, w, c = img.shape
    X = img.reshape(-1, c).astype(np.float32)
    n_components = max(1, min(n_components, c))
    pca = PCA(n_components=n_components, svd_solver='auto')
    Z = pca.fit_transform(X)
    Xr = pca.inverse_transform(Z)
    recon = np.clip(Xr.reshape(h, w, c), 0, 255).astype(np.uint8)
    explained = float(np.sum(pca.explained_variance_ratio_))
    return recon, explained


def texture_glcm(img: np.ndarray):
    if graycomatrix is None or graycoprops is None or rgb2gray is None:
        return None
    gray = img if img.ndim == 2 else (rgb2gray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) * 255).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1, 2, 4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    feats = {
        'contrast': np.mean(graycoprops(glcm, 'contrast')),
        'dissimilarity': np.mean(graycoprops(glcm, 'dissimilarity')),
        'homogeneity': np.mean(graycoprops(glcm, 'homogeneity')),
        'ASM': np.mean(graycoprops(glcm, 'ASM')),
        'energy': np.mean(graycoprops(glcm, 'energy')),
        'correlation': np.mean(graycoprops(glcm, 'correlation')),
    }
    return feats


def ransac_line_fit(img: np.ndarray) -> np.ndarray:
    if ransac is None or LineModelND is None:
        return img
    edges = canny(to_gray(img)) if canny is not None else (canny_edges(img, 100, 200) > 0)
    yx = np.column_stack(np.nonzero(edges))  # (row, col)
    if len(yx) < 2:
        return img
    model, inliers = ransac(yx, LineModelND, min_samples=2, residual_threshold=1, max_trials=1000)
    out = img.copy()
    if model is not None:
        # draw fitted line
        (r0, c0), (r1, c1) = model.params
        p0 = (int(c0), int(r0))
        p1 = (int(c1), int(r1))
        cv2.line(out, p0, p1, (255, 0, 0), 2)
    return out


def contour_shapes(img: np.ndarray, min_area: float = 100.0) -> np.ndarray:
    bw = threshold_binary(img, method="Otsu")
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = 0.02 * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, f"n={len(approx)}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out

###############################################################################
# Streamlit UI
###############################################################################

st.set_page_config(
    page_title="Pro Vision Lab – Image Processor",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .css-18ni7ap, .stMarkdown p {font-size: 0.98rem;}
    .st-emotion-cache-1r4qj8v {border-radius: 1rem !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🖼️ Pro Vision Lab – Image Processor")
st.caption("Computer Vision | Filtering | Features | Segmentation | PCA | RANSAC | Color Spaces")

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff"])
    demo = st.checkbox("Use demo image", value=False, help="Use a built-in sample if no file is uploaded")
    max_w = st.slider("Max display width", 400, 1600, 900, 50)

# Load image
if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
elif demo:
    # generate a simple demo image
    demo_img = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(demo_img, (150, 120), 60, (0, 255, 0), -1)
    cv2.rectangle(demo_img, (280, 60), (420, 200), (255, 0, 0), 3)
    cv2.putText(demo_img, "Pro Vision", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    pil_img = cv_to_pil(demo_img)
else:
    st.info("Upload an image or enable 'Use demo image' in the sidebar.")
    st.stop()

orig_cv = pil_to_cv(pil_img)

# Tabs for categories
basic_tab, filter_tab, edges_tab, binary_tab, seg_tab, feat_tab, shape_tab, pca_tab, model_tab, tesla_tab = st.tabs([
    "Basic", "Filtering", "Edges", "Binary/Morph", "Segmentation", "Features", "Shape", "PCA", "Model Fitting", "Tesla Case Study"
])

with basic_tab:
    st.subheader("Basic transforms & color spaces")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Resize**")
        w = st.number_input("Width (px)", min_value=0, value=0, step=32)
        h = st.number_input("Height (px)", min_value=0, step=32)
        basic_img = orig_cv.copy()
        if w > 0 or h > 0:
            basic_img = resize_image(basic_img, width=(w or None), height=(h or None))
        st.image(cv_to_pil(ensure_uint8(basic_img)), caption="Resized", use_column_width=True)
    with col2:
        st.markdown("**Color Space**")
        space = st.selectbox("Target space", ["RGB", "HSV", "LAB", "YCrCb"])
        cs_img = change_colorspace(orig_cv, space)
        st.image(cv_to_pil(ensure_uint8(cs_img)), caption=f"Converted to {space}", use_column_width=True)
    with col3:
        st.markdown("**Gemini Info**")
        st.info(get_gemini_explanation("Resize"))
        st.info(get_gemini_explanation(f"Color Space: {space}"))

with filter_tab:
    st.subheader("Image filtering")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        k = st.slider("Gaussian kernel size", 1, 25, 5, step=2)
        sigma = st.slider("Sigma", 0.0, 10.0, 1.0, 0.1)
        g = gaussian_blur(orig_cv, k, sigma)
        st.image(cv_to_pil(g), caption="Gaussian blur")
    with fcol2:
        km = st.slider("Median kernel size", 1, 25, 5, step=2)
        m = median_blur(orig_cv, km)
        st.image(cv_to_pil(m), caption="Median blur")
    with fcol3:
        d = st.slider("Bilateral diameter", 1, 25, 9)
        sc = st.slider("SigmaColor", 1, 150, 75)
        ss = st.slider("SigmaSpace", 1, 150, 75)
        b = bilateral_filter(orig_cv, d, sc, ss)
        st.image(cv_to_pil(b), caption="Bilateral filter")

with edges_tab:
    st.subheader("Edge detection")
    e1, e2 = st.columns(2)
    with e1:
        sob = sobel_edges(orig_cv)
        st.image(cv_to_pil(sob), caption="Sobel magnitude")
    with e2:
        t1 = st.slider("Canny threshold1", 0, 255, 100)
        t2 = st.slider("Canny threshold2", 0, 255, 200)
        can = canny_edges(orig_cv, t1, t2)
        st.image(cv_to_pil(can), caption="Canny edges")

with binary_tab:
    st.subheader("Binary images & morphology")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        method = st.selectbox("Threshold method", ["Otsu", "Adaptive (mean)", "Adaptive (gaussian)", "Manual"])
        thr = st.slider("Manual threshold", 0, 255, 127)
        bw = threshold_binary(orig_cv, method, thr)
        st.image(cv_to_pil(bw), caption=f"Binary – {method}")
    with bcol2:
        op = st.selectbox("Morphology op", ["erode", "dilate", "open", "close"]) 
        ksz = st.slider("Kernel size", 1, 25, 3)
        it = st.slider("Iterations", 1, 10, 1)
        mor = morphology(bw, op, ksz, it)
        st.image(cv_to_pil(mor), caption=f"Morphology – {op}")

with seg_tab:
    st.subheader("Segmentation")
    scol1, scol2 = st.columns(2)
    with scol1:
        k = st.slider("KMeans K", 2, 10, 3)
        km_img = kmeans_segmentation(orig_cv, k)
        st.image(cv_to_pil(km_img), caption=f"KMeans (K={k})")
    with scol2:
        nseg = st.slider("SLIC segments", 50, 1000, 300, 50)
        comp = st.slider("Compactness", 1.0, 30.0, 10.0)
        sp = slic_superpixels(orig_cv, nseg, comp)
        st.image(cv_to_pil(sp), caption="SLIC superpixels (with boundaries)")

with feat_tab:
    st.subheader("Feature detection & descriptors")
    f1, f2, f3 = st.columns(3)
    with f1:
        harris = harris_corners(orig_cv)
        st.image(cv_to_pil(harris), caption="Harris corners")
    with f2:
        shi = shi_tomasi(orig_cv)
        st.image(cv_to_pil(shi), caption="Shi–Tomasi corners")
    with f3:
        kf = st.slider("Max features", 100, 2000, 500, 100)
        feat = sift_orb_features(orig_cv, kf)
        st.image(cv_to_pil(feat), caption="SIFT (fallback to ORB)")

with shape_tab:
    st.subheader("Shape analysis (contours)")
    min_area = st.slider("Min area", 10, 5000, 200)
    sh = contour_shapes(orig_cv, float(min_area))
    st.image(cv_to_pil(sh), caption="Contours + polygon approx + bounding boxes")

with pca_tab:
    st.subheader("PCA – Dimensionality reduction on colors")
    comps = st.slider("Components", 1, 3, 2)
    pr, exp = pca_compress(orig_cv, n_components=comps)
    st.image(cv_to_pil(pr), caption=f"PCA reconstruction (explained variance ≈ {exp*100:.1f}%)")

with model_tab:
    st.subheader("Probabilistic model fitting (RANSAC line)")
    ran = ransac_line_fit(orig_cv)
    st.image(cv_to_pil(ran), caption="RANSAC-fitted line on edges")

with tesla_tab:
    st.subheader("Case Study – How Tesla uses Cameras & CV for Autopilot/FSDS")
    st.markdown(
        """
        **High-level pipeline:** Multi-camera surround views (up to 8+ cameras) feed synchronized frames into a neural perception stack. Key components:
        
        1. **Image formation & calibration**: Lens distortion correction, color calibration, rolling-shutter handling, and precise camera extrinsics to align views in a unified vehicle-centric frame.
        2. **Filtering & edges**: Classical CV still helps—denoising, edge cues and corners can refine proposals and improve stability in low-light or high-noise frames.
        3. **Features & tracking**: Corners/keypoints for geometric consistency, plus descriptors for short-term tracking when learning-based trackers are uncertain.
        4. **Segmentation & detection**: Deep nets segment drivable space, lane markings, and detect vehicles/pedestrians/cyclists. Superpixels/region proposals can act as priors.
        5. **3D understanding**: Multi-view geometry + neural networks infer depth and build **occupancy networks** to represent free space and obstacles around the car.
        6. **Model fitting**: Curves/lanes are fit with polynomials or splines; **RANSAC**-style robust estimators reject outliers (shadows, occlusions, debris).
        7. **Probabilistic fusion**: Outputs become a **probabilistic scene graph**, fused over time (Bayesian filters) to yield stable trajectories and intent predictions.
        
        **Why this matters here:** This app demonstrates the classical building blocks—filtering, edges, corners, segmentation, PCA, and robust model fitting—that underpin or support modern deep pipelines. Try combining Canny → RANSAC to see robust line fits reminiscent of lane estimation.
        """
    )

st.divider()

st.markdown(
    """
    **Tips**  
    • Use *Filtering* to denoise before edges/thresholds.  
    • Use *Binary/Morph* to clean up masks before *Shape* analysis.  
    • If SIFT is unavailable in your OpenCV build, the app falls back to ORB automatically.  
    • Superpixels require `scikit-image`. PCA requires `scikit-learn`.
    """
)

st.caption("Built with Streamlit, OpenCV, and scikit-image/sklearn when available. © 2025")
