import streamlit as st
import os
from PIL import Image
import json
from typing import Dict, List, Optional
import uuid
import time
from datetime import datetime


class AnnotationInterface:
    def __init__(self):
        # Configuration
        self.image_base_path = "data/images/sample"
        self.ground_truth_base_path = "data/raw-ocr/truth"
        self.ocr_base_path = "data/raw-ocr/judged/context-32b-candidate-3"

        # Ensure directories exist
        os.makedirs(self.ground_truth_base_path, exist_ok=True)

        # Initialize session state
        if 'current_annotation' not in st.session_state:
            st.session_state.current_annotation = {}
        if 'book_list' not in st.session_state:
            st.session_state.book_list = self._get_book_list()
        if 'reset_trigger' not in st.session_state:
            st.session_state.reset_trigger = False
        if 'current_text' not in st.session_state:
            st.session_state.current_text = ""

    def _get_book_list(self) -> List[str]:
        """Get list of available books"""
        try:
            books = [d for d in os.listdir(self.image_base_path)
                     if os.path.isdir(os.path.join(self.image_base_path, d))]
            return sorted(books)
        except FileNotFoundError:
            st.error(f"Image directory not found: {self.image_base_path}")
            return []

    def _get_book_images(self, book_name: str) -> List[str]:
        """Get list of images for a specific book"""
        book_path = os.path.join(self.image_base_path, book_name)
        try:
            images = [f for f in os.listdir(book_path)
                      if f.lower().endswith(('.jp2', '.jpeg', '.jpg', '.png'))]
            return sorted(images)
        except FileNotFoundError:
            st.error(f"Book directory not found: {book_path}")
            return []

    def _get_ground_truth_path(self, book_name: str, image_name: str) -> str:
        """Get the path for ground truth file"""
        # Remove extension from image name and add .txt
        base_name = os.path.splitext(image_name)[0]
        book_truth_path = os.path.join(self.ground_truth_base_path, book_name)
        os.makedirs(book_truth_path, exist_ok=True)
        return os.path.join(book_truth_path, f"{base_name}.txt")

    def _load_ground_truth(self, book_name: str, image_name: str) -> Optional[str]:
        """Load ground truth text if it exists"""
        truth_path = self._get_ground_truth_path(book_name, image_name)
        try:
            with open(truth_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None

    def _get_file_metadata(self, book_name: str, image_name: str) -> Optional[Dict]:
        """Get creation and modification times for ground truth file"""
        truth_path = self._get_ground_truth_path(book_name, image_name)
        if not os.path.exists(truth_path):
            return None

        try:
            created = os.path.getctime(truth_path)
            modified = os.path.getmtime(truth_path)

            return {
                'created': datetime.fromtimestamp(created).strftime('%Y-%m-%d %H:%M:%S'),
                'modified': datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S'),
                'size': os.path.getsize(truth_path)
            }
        except OSError:
            return None

    def _save_ground_truth(self, book_name: str, image_name: str, text: str) -> None:
        """Save ground truth text"""
        truth_path = self._get_ground_truth_path(book_name, image_name)
        with open(truth_path, 'w', encoding='utf-8') as f:
            f.write(text)

    def _load_ocr_result(self, book_name: str, image_name: str) -> Optional[str]:
        """Load OCR result if it exists"""
        # This would typically come from your OCR system
        # For now, we'll check if there's a corresponding text file
        image_path = os.path.join(self.ocr_base_path, book_name, image_name)
        ocr_path = os.path.splitext(image_path)[0] + '.txt'

        try:
            with open(ocr_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None

    def _load_image(self, book_name: str, image_name: str):
        """Load and display image"""
        image_path = os.path.join(self.image_base_path, book_name, image_name)

        try:
            if image_name.lower().endswith('.jp2'):
                # JP2 files might need special handling
                # You might need to install additional libraries like glymur
                st.warning(
                    "JP2 format may require special handling. Consider converting to JPEG/PNG for better compatibility.")

            # Try to open with PIL
            image = Image.open(image_path)
            return image
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None

    def render(self):
        st.header("OCR Annotation Tool")

        # Book selection
        if not st.session_state.book_list:
            st.error("No books found. Please check your image directory configuration.")
            return

        selected_book = st.selectbox(
            "Select a Book",
            options=st.session_state.book_list,
            index=0,
            key="book_selector"
        )

        # Image selection
        book_images = self._get_book_images(selected_book)
        if not book_images:
            st.error(f"No images found in {selected_book} directory.")
            return

        selected_image = st.selectbox(
            "Select an Image",
            options=book_images,
            index=0,
            key="image_selector"
        )

        # Load OCR result and ground truth
        ocr_text = self._load_ocr_result(selected_book, selected_image) or ""
        ground_truth = self._load_ground_truth(selected_book, selected_image)
        file_metadata = self._get_file_metadata(selected_book, selected_image)

        # Initialize or update current text
        image_key = f"{selected_book}_{selected_image}"
        if image_key not in st.session_state:
            st.session_state[image_key] = ground_truth if ground_truth else ocr_text

        # Handle reset functionality
        if st.session_state.reset_trigger:
            st.session_state[image_key] = ocr_text
            st.session_state.reset_trigger = False
            st.rerun()

        # Display image and annotation interface
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Image Preview")
            image = self._load_image(selected_book, selected_image)
            if image:
                st.image(image, use_container_width=True)

            # Image metadata
            st.caption(f"Book: {selected_book} | Image: {selected_image}")
            if image:
                st.caption(f"Size: {image.size[0]}x{image.size[1]} | Format: {image.format}")

        with col2:
            st.subheader("Annotation")

            if ocr_text:
                st.info("OCR Result Loaded")
            else:
                st.warning("No OCR result found for this image")

            # Text area for annotation with a unique key per image
            annotated_text = st.text_area(
                "Edit OCR Result",
                value=st.session_state[image_key],
                height=400,
                help="Correct the OCR text to create ground truth data",
                key=f"text_editor_{image_key}"
            )

            # Update the stored text
            st.session_state[image_key] = annotated_text

            # Annotation controls
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Save Annotation", type="primary", use_container_width=True):
                    self._save_ground_truth(selected_book, selected_image, annotated_text)
                    # Refresh metadata after saving
                    file_metadata = self._get_file_metadata(selected_book, selected_image)
                    st.success("Annotation saved successfully!")
                    # Small delay to show success message before rerun
                    time.sleep(0.5)
                    st.rerun()

            with col_b:
                if st.button("Reset to OCR", use_container_width=True):
                    st.session_state.reset_trigger = True
                    st.rerun()

            # Annotation statistics
            st.subheader("Annotation Stats")

            if file_metadata:
                st.metric("Status", "Annotated", help="This image has been annotated")

                # Display file metadata
                # with st.expander("File Details", expanded=False):
                st.write(f"Created: {file_metadata['created']}")
                st.write(f"Last Modified: {file_metadata['modified']}")
                # st.write(f"File Size: {file_metadata['size']} bytes")

                # Calculate character difference
                if ocr_text:
                    char_diff = len(annotated_text) - len(ocr_text)
                    st.metric("Character Difference", char_diff)
            else:
                st.metric("Status", "Not Annotated", help="This image hasn't been annotated yet")
                if ocr_text:
                    st.metric("OCR Text Length", len(ocr_text))

        # Additional tools
        st.subheader("Annotation Tools")

        tab1, tab2, tab3 = st.tabs(["Quality Metrics", "Batch Operations", "Export"])

        with tab1:
            st.write("OCR Quality Evaluation")
            if ocr_text and ground_truth:
                # Calculate basic metrics
                ocr_len = len(ocr_text)
                truth_len = len(ground_truth)
                diff = abs(ocr_len - truth_len)

                col1, col2, col3 = st.columns(3)
                col1.metric("OCR Length", ocr_len)
                col2.metric("Truth Length", truth_len)
                col3.metric("Length Difference", diff)

                # Simple character accuracy (this is a placeholder)
                # In a real system, you'd use more sophisticated metrics like CER, WER
                matching_chars = sum(1 for a, b in zip(ocr_text, ground_truth) if a == b)
                min_len = min(len(ocr_text), len(ground_truth))
                accuracy = (matching_chars / min_len * 100) if min_len > 0 else 0

                st.metric("Character Accuracy", f"{accuracy:.1f}%")
            else:
                st.info("Need both OCR and ground truth to calculate metrics")

        with tab2:
            st.write("Batch Annotation Operations")
            st.info("This feature would allow batch processing of images")

            if st.button("Export All Annotations"):
                self._export_all_annotations()

        with tab3:
            st.write("Export Options")
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "TXT", "CSV"]
            )

            if st.button("Export Current Book"):
                self._export_book_annotations(selected_book, export_format)

    def get_annotation_stats(self):
        """Calculate annotation statistics"""
        books = self._get_book_list()

        total_images = 0
        annotated_images = 0

        for book in books:
            images = self._get_book_images(book)
            total_images += len(images)
            for image in images:
                if self._load_ground_truth(book, image) is not None:
                    annotated_images += 1

        return {
            "total_books": len(books),
            "total_images": total_images,
            "annotated_images": annotated_images,
            "remaining_images": total_images - annotated_images,
            "completion_percentage": (annotated_images / total_images * 100) if total_images > 0 else 0
        }
    def _export_all_annotations(self):
        """Export all annotations to a single file"""
        # This would iterate through all books and images
        st.info("Export feature would be implemented here")

    def _export_book_annotations(self, book_name: str, format: str):
        """Export annotations for a specific book"""
        # This would export all annotations for the selected book
        st.info(f"Would export {book_name} in {format} format")