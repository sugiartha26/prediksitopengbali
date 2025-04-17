import streamlit as st 
import numpy as np
import tensorflow as tf
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="topeng_model_quant.tflite")
interpreter.allocate_tensors()

# Ambil detail tensor input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kelas topeng Bali
class_names = ['keras', 'penasar', 'wijil', 'tua', 'bujuh', 'dalem', 'sidakarya']

# Preprocessing gambar
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    img_array = np.array(image).astype(np.float32)
    if img_array.shape[-1] == 4:  # RGBA ke RGB
        img_array = img_array[:, :, :3]
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Fungsi prediksi dengan TFLite
def predict_with_tflite(image_array: np.ndarray) -> (str, float):
    # Sesuaikan dtype input
    input_data = image_array.astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    confidence = float(np.max(output_data)) * 100
    return class_names[predicted_index], confidence

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Topeng Bali",
    page_icon="🎭",
    layout="centered"
)

# Sidebar menu
with st.sidebar:
    st.header("📂 Menu")
    menu = st.radio("Pilih halaman:", ["Penjelasan", "Prediksi"])

# ---------------- PENJELASAN ----------------
if menu == "Penjelasan":
    st.title("🎭 Klasifikasi Topeng Bali")
    st.markdown("""
    Aplikasi ini menggunakan model **Deep Learning** untuk mengklasifikasikan gambar **topeng tradisional Bali** ke dalam kategori tertentu berdasarkan karakteristik visual.

    ---
    ### 🧾 Jenis Topeng yang Didukung:
    - 🟥 **Keras** – Tokoh ksatria gagah.
    - 🟡 **Penasar** – Narator jenaka.
    - 🟢 **Wijil** – Pembantu polos.
    - 🟤 **Tua** – Orang tua bijak.
    - ⚫ **Bujuh** – Karakter aneh.
    - ⚪ **Dalem** – Raja utama.
    - 🟣 **Sidakarya** – Topeng sakral.

    ---
    📱 *Untuk pengguna mobile:*  
    Gunakan tombol ☰ di kiri atas untuk membuka menu.
    """)

# ---------------- PREDIKSI ----------------
elif menu == "Prediksi":
    st.title("🔍 Prediksi Topeng Bali")
    st.write("Unggah gambar topeng Bali Anda dan lihat jenis topengnya berdasarkan klasifikasi model.")

    uploaded_file = st.file_uploader("📷 Pilih Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Gambar yang Diunggah", width=280)

        with st.spinner("⏳ Memproses gambar..."):
            processed_image = preprocess_image(image)
            predicted_class, confidence = predict_with_tflite(processed_image)

        st.markdown("---")
        st.subheader("📌 Hasil Prediksi")
        st.success(f"**Jenis Topeng: {predicted_class.capitalize()}**")
        st.write(f"🎯 Keyakinan model: **{confidence:.2f}%**")

        st.image(Image.fromarray((processed_image[0] * 255).astype(np.uint8)),
                 caption="Gambar Setelah Resize (224x224)", width=224)

        st.markdown("---")
        st.info("💡 Tips: Gunakan gambar dengan pencahayaan baik dan posisi topeng yang jelas.")
