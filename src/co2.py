import streamlit as st
import numpy as np
from PIL import Image
import cv2
from sklearn.decomposition import PCA
import io
import google.generativeai as genai

@st.cache_data(show_spinner=False) 
def get_image_info(filter_name):
    # --- FIX: Use API Key directly ---    
<<<<<<< HEAD
    API_KEY = "YOUR_API_KEY_HERE"
=======
    API_KEY = "AIzaSyCaZqn1vSnAhqxs6OLCNYZjN53YBgdqQFs"
>>>>>>> d7baaaf (t)
    if API_KEY == "YOUR_API_KEY_HERE":
        st.warning("Please add your Gemini API key to the code to enable this feature.", icon="⚠️")
        return "API key not provided. Please add it to the `get_image_info` function in the code."

    try:
        genai.configure(api_key=API_KEY)        
        model = genai.GenerativeModel('gemini-1.5-flash-latest')        
        prompt = f"In 2-3 sentences, explain what the '{filter_name}' image processing technique does. Frame it for a user of a photo editing application."        
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:        
        st.error(f"Could not connect to Gemini API. Please check your API key. Error: {e}", icon="🔑")
        return "Information could not be retrieved. Please check your API key configuration."

def convert_image(img):    
    buf = io.BytesIO()    
    if img.mode == 'L':
        img.save(buf, format="PNG")
    else:
        img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def process_image(image, operation, **kwargs):        
    img_array = np.array(image)    
    if len(img_array.shape) == 2: # Grayscale input
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4: # RGBA input
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # --- Image Enhancement ---
    if operation == "Grayscale":
        processed_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    elif operation == "Brightness":
        value = kwargs.get('value', 30)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        processed_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    elif operation == "Contrast":
        alpha = kwargs.get('value', 1.5) # Contrast control
        beta = 0 
        processed_img = cv2.convertScaleAbs(img_array, alpha=alpha, beta=beta)
    elif operation == "Gaussian Blur (Low Pass)":
        ksize = kwargs.get('ksize', (15, 15))
        processed_img = cv2.GaussianBlur(img_array, ksize, 0)
    elif operation == "High Pass Filter":
        blurred = cv2.GaussianBlur(img_array, (21, 21), 0)
        processed_img = cv2.addWeighted(img_array, 1.5, blurred, -0.5, 0)
    elif operation == "Invert":
        processed_img = cv2.bitwise_not(img_array)

    # --- Image Restoration ---
    elif operation == "Median Filter":
        ksize = kwargs.get('ksize', 5)
        processed_img = cv2.medianBlur(img_array, ksize)
    elif operation == "Denoising":
        processed_img = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)

    # --- Image Segmentation ---
    elif operation == "Thresholding":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, processed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    elif operation == "Otsu's Binarization":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Image Compression ---
    elif operation == "JPEG Compression":
        quality = kwargs.get('quality', 90)
        pil_img = Image.fromarray(img_array)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        processed_img_pil = Image.open(buf)
        return processed_img_pil

    # --- Image Synthesis ---
    elif operation == "Generate Noise":
        noise = np.random.randint(0, 255, img_array.shape, dtype=np.uint8)
        processed_img = cv2.add(img_array, noise)

    # --- Edge Detection ---
    elif operation == "Canny Edge Detection":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.Canny(gray, 100, 200)
    elif operation == "Sobel Edge Detection":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        processed_img = cv2.magnitude(sobelx, sobely)

    # --- PCA ---
    elif operation == "Principal Component Analysis (PCA)":
        n_components = kwargs.get('n_components', 50)
        blue, green, red = cv2.split(img_array)
        pca_b = PCA(n_components=n_components)
        pca_g = PCA(n_components=n_components)
        pca_r = PCA(n_components=n_components)
        trans_pca_b = pca_b.fit_transform(blue)
        trans_pca_g = pca_g.fit_transform(green)
        trans_pca_r = pca_r.fit_transform(red)
        recon_pca_b = pca_b.inverse_transform(trans_pca_b)
        recon_pca_g = pca_g.inverse_transform(trans_pca_g)
        recon_pca_r = pca_r.inverse_transform(trans_pca_r)
        
        # --- FIX: Clip values to the valid 0-255 range and convert type ---        
        recon_pca_b = np.clip(recon_pca_b, 0, 255)
        recon_pca_g = np.clip(recon_pca_g, 0, 255)
        recon_pca_r = np.clip(recon_pca_r, 0, 255)        
        processed_img = cv2.merge((
            recon_pca_b.astype(np.uint8), 
            recon_pca_g.astype(np.uint8), 
            recon_pca_r.astype(np.uint8)
        ))

    # --- Corner Detection ---
    elif operation == "Harris Corner Detection":
        processed_img = img_array.copy()
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)    
        corners = np.argwhere(dst > 0.01 * dst.max())
        for corner in corners:
            y, x = corner
            cv2.circle(processed_img, (x, y), 3, (255, 0, 0), -1)
    elif operation == "Shi-Tomasi Corner Detection":
        processed_img = img_array.copy()
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()            
            cv2.circle(processed_img, (x, y), 3, (255, 0, 0), -1)

    # --- Feature Extraction ---
    elif operation == "SIFT (Scale-Invariant Feature Transform)":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        keypoints, _ = sift.detectAndCompute(gray, None)
        processed_img = cv2.drawKeypoints(gray, keypoints, img_array.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif operation == "SURF (Speeded-Up Robust Features)":
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            surf = cv2.xfeatures2d.SURF_create(400)
            keypoints, _ = surf.detectAndCompute(gray, None)
            processed_img = cv2.drawKeypoints(gray, keypoints, img_array.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        except (cv2.error, AttributeError):
            st.error("SURF is not available in your OpenCV version. It's part of the patented algorithms in `opencv-contrib-python`. Please install a compatible version.")
            return image 
    else:
        return image 
    return Image.fromarray(processed_img)


st.set_page_config(page_title="Gemini Image Editor", layout="wide")
st.title("🖼️ Multi-Page Image Editor")
st.text("A comprehensive tool for image processing powered by Python and Streamlit.")

# --- Sidebar for Navigation and Controls ---
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        [
            "Image Enhancement", "Image Restoration", "Image Segmentation",
            "Image Compression", "Image Synthesis", "Edge Detection",
            "Principal Component Analysis", "Corner Detection", "Feature Extraction"
        ]
    )
    st.markdown("---")
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    st.info("This app uses the Gemini API. Please add your API key directly in the python script to enable the info feature.")

# --- Main Page Content ---
if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")        
    if page == "Image Enhancement":
        st.header("✨ Image Enhancement")
        operation = st.selectbox(
            "Choose an enhancement technique",
            ["Brightness", "Contrast", "Grayscale", "Gaussian Blur (Low Pass)", "High Pass Filter", "Invert"]
        )
        
        kwargs = {}
        if operation == "Brightness":
            kwargs['value'] = st.slider("Brightness Level", -100, 100, 30)
        elif operation == "Contrast":
            kwargs['value'] = st.slider("Contrast Level", 1.0, 3.0, 1.5)
        elif operation == "Gaussian Blur (Low Pass)":
            k_size = st.slider("Kernel Size", 1, 31, 15, step=2)
            kwargs['ksize'] = (k_size, k_size)

    elif page == "Image Restoration":
        st.header("🔧 Image Restoration")
        operation = st.selectbox(
            "Choose a restoration technique",
            ["Median Filter", "Denoising"]
        )
        kwargs = {}
        if operation == "Median Filter":
            k_size = st.slider("Kernel Size", 1, 15, 5, step=2)
            kwargs['ksize'] = k_size
            
    elif page == "Image Segmentation":
        st.header("🎨 Image Segmentation")
        operation = st.selectbox(
            "Choose a segmentation technique",
            ["Thresholding", "Otsu's Binarization"]
        )
        kwargs = {}

    elif page == "Image Compression":
        st.header("🗜️ Image Compression")
        operation = "JPEG Compression"
        kwargs = {}
        kwargs['quality'] = st.slider("JPEG Quality", 0, 100, 90)

    elif page == "Image Synthesis":
        st.header("⚗️ Image Synthesis")
        operation = "Generate Noise"
        kwargs = {}

    elif page == "Edge Detection":
        st.header("🔪 Edge Detection")
        operation = st.selectbox(
            "Choose an edge detection algorithm",
            ["Canny Edge Detection", "Sobel Edge Detection"]
        )
        kwargs = {}

    elif page == "Principal Component Analysis":
        st.header("📊 Principal Component Analysis (PCA)")
        operation = "Principal Component Analysis (PCA)"
        kwargs = {}
        max_components = min(original_image.size[0], original_image.size[1], 300)
        kwargs['n_components'] = st.slider("Number of Principal Components", 1, max_components, 50)

    elif page == "Corner Detection":
        st.header("📐 Corner Detection")
        operation = st.selectbox(
            "Choose a corner detection algorithm",
            ["Harris Corner Detection", "Shi-Tomasi Corner Detection"]
        )
        kwargs = {}

    elif page == "Feature Extraction":
        st.header("🌟 Feature Extraction")
        st.warning("Note: SURF may require a specific version of `opencv-contrib-python`.")
        operation = st.selectbox(
            "Choose a feature extraction algorithm",
            ["SIFT (Scale-Invariant Feature Transform)", "SURF (Speeded-Up Robust Features)"]
        )
        kwargs = {}

    else:
        st.error("Page not found!")
        operation = None
        kwargs = {}

    # --- Display Images and Info ---
    if operation:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_container_width=True)

        with col2:
            st.subheader("Processed Image")
            with st.spinner("Applying filter..."):
                processed_image = process_image(original_image, operation, **kwargs)
                st.image(processed_image, use_container_width=True)

        # --- Info Box ---
        st.markdown("---")
        st.subheader(f"ℹ️ About: {operation}")
        with st.expander("Click to learn more", expanded=True):
            with st.spinner("Asking Gemini for info..."):
                info_text = get_image_info(operation)
                st.info(info_text)
        
        st.markdown("---")
        st.download_button(
            label="Download Processed Image",
            data=convert_image(processed_image),
            file_name=f"processed_{operation.lower().replace(' ', '_')}.png",
            mime="image/png"
        )

else:
    st.info("Please upload an image using the sidebar to get started.")