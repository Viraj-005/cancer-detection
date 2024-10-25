import streamlit as st
from PIL import Image, ImageDraw

def app():
    # Function to round the corners of the image
    def round_corners(image, radius=15):
        width, height = image.size
        rounded_mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(rounded_mask)
        draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)

        # Create a new image with transparency and paste the original image onto it
        rounded_image = Image.new('RGBA', (width, height))
        rounded_image.paste(image, (0, 0), rounded_mask)
        return rounded_image

    # Load the image
    image_path = 'images/CancerDetective.png'
    image = Image.open(image_path)

    # Resize the image while keeping the height constant
    new_width = 1200
    new_height = 500
    resized_image = image.resize((new_width, new_height))

    # Round the corners of the image
    rounded_image = round_corners(resized_image, radius=15)

    # Display the resized and rounded image
    st.image(rounded_image, use_column_width=True)

    st.markdown("""
                <div class="main">
                    <h2>🔬 Welcome to Cancer Detective Web App 🎗️</h2>
                    <p>
                        Our web app harnesses the power of AI 🧠 to assist in the early detection of cancer. 
                        Focused on leukemia, lung, and skin cancers, Cancer Detective analyzes medical 
                        images 📷 to provide reliable predictions, helping doctors 👩‍⚕️👨‍⚕️ and patients 🧑‍⚕️ make informed decisions. 
                        Early detection can save lives ❤️, and our goal is to make this technology accessible 🌍 and easy to use for all.
                    </p>
                    <div class="feature-box">
                        <h2>✨ Key Features:</h2>
                        <ul>
                            <li>🎯 <strong>Accurate Detection</strong>: Leveraging deep learning models to detect cancerous images, tissues with precision.</li>
                            <li>🖱️ <strong>User-Friendly</strong>: Simple interface designed for ease of use, making advanced AI accessible.</li>
                            <li>⚡ <strong>Fast Results</strong>: Quick processing times for image analysis and diagnosis.</li>
                            <li>🩺 <strong>Multiple Cancer Types</strong>: Support for leukemia, lung cancer, and skin cancer detection.</li>
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)

