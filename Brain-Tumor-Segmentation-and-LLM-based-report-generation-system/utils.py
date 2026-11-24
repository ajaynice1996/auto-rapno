# segmentation_pipeline.py
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, MaxPool2D, Conv2DTranspose, concatenate
from skimage.measure import label, regionprops
import requests
import json


class TumorSegmentationModel:
    def __init__(self, model_path, input_size=(256, 256, 3)):
        self.model = self.build_unet(input_size)
        self.model.load_weights(model_path)
        self.input_size = input_size[:2]

    def build_unet(self, input_size):
        def conv_block(inputs, n_filters, max_pool=True):
            x = Conv2D(n_filters, 3, padding='same', kernel_initializer='he_normal', use_bias=False)(inputs)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)
            x = Conv2D(n_filters, 3, padding='same', kernel_initializer='he_normal', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)
            skip = x
            next_layer = MaxPool2D()(x) if max_pool else x
            return next_layer, skip

        def up_block(reg_inputs, skip_inputs, n_filters):
            x = Conv2DTranspose(n_filters, 3, 2, padding='same')(reg_inputs)
            x = concatenate([x, skip_inputs], axis=3)
            x = conv_block(x, n_filters, max_pool=False)[0]
            return x

        inputs = Input(shape=input_size, name='Input')
        cb1, cb1_skip = conv_block(inputs, 32)
        cb2, cb2_skip = conv_block(cb1, 64)
        cb3, cb3_skip = conv_block(cb2, 128)
        cb4, cb4_skip = conv_block(cb3, 256)
        cb5 = conv_block(cb4, 512, max_pool=False)
        up1 = up_block(cb5[0], cb4_skip, 256)
        up2 = up_block(up1, cb3_skip, 128)
        up3 = up_block(up2, cb2_skip, 64)
        up4 = up_block(up3, cb1_skip, 32)
        conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', use_bias=False)(up4)
        bn = BatchNormalization()(conv1)
        act = LeakyReLU(0.2)(bn)
        outputs = Conv2D(1, 1, padding='same', activation='sigmoid', name='Output')(act)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='mean_squared_error')
        return model

    def preprocess_image(self, image_path_or_bytes):
        if isinstance(image_path_or_bytes, str):
            img = Image.open(image_path_or_bytes).convert('RGB')
        else:
            img = Image.open(image_path_or_bytes).convert('RGB')

        img_resized = img.resize(self.input_size)
        img_array = np.array(img_resized) / 255.0
        return np.expand_dims(img_array, axis=0), np.array(img_resized)

    def predict_mask(self, input_tensor):
        pred_mask = self.model.predict(input_tensor)[0, :, :, 0]
        binary_mask = (pred_mask > 0.5).astype(np.uint8)
        return binary_mask

    def overlay_mask_on_image(self, image, mask, alpha=0.6, mask_color=(255, 0, 0), border_color=(0, 0, 200), border_thickness=1):
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        color_mask = np.zeros_like(image)
        color_mask[mask == 1] = mask_color

        overlayed = cv2.addWeighted(image, 1.0, color_mask, alpha, 0)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlayed, contours, -1, border_color, thickness=border_thickness)
        return overlayed
    

class TumorReportGenerator:
    def __init__(self):
        pass

    def extract_tumor_features(self, mask: np.ndarray, image_shape: tuple) -> dict:
        """
        Extracts tumor features from the binary mask.
        
        Parameters:
            mask (np.ndarray): 2D binary numpy array of predicted tumor mask (1 = tumor, 0 = background)
            image_shape (tuple): shape of the original image (height, width)
            
        Returns:
            dict: features including detection, area, location, shape, bounding box
        """

        features = {}
        tumor_pixels = np.sum(mask)
        total_pixels = image_shape[0] * image_shape[1]
        area_percent = (tumor_pixels / total_pixels) * 100
        features['tumor_detected'] = tumor_pixels > 0

        if not features['tumor_detected']:
            features.update({
                'area': 0,
                'location': 'No tumor detected',
                'shape': 'N/A',
                'bounding_box': None
            })
            return features

        features['area'] = int(area_percent)
        features['area_percent'] = round(area_percent, 2)
        features['confidence'] = round(mask.max() * 100, 2)


        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)
        features['bounding_box'] = (x, y, x + w, y + h)

        center_x = x + w // 2
        center_y = y + h // 2
        height, width = image_shape[:2]

        if center_x < width // 2:
            horizontal = 'Left'
        else:
            horizontal = 'Right'

        if center_y < height // 3:
            vertical = 'Frontal lobe'
        elif center_y < 2 * height // 3:
            vertical = 'Temporal lobe'
        else:
            vertical = 'Occipital lobe'

        features['location'] = f"{horizontal} {vertical}"

        contour_area = cv2.contourArea(largest_contour)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0

        features['shape'] = 'Regular' if solidity > 0.9 else 'Irregular'

        return features

    def generate_llm_report(self, features: dict) -> str:
        if not features['tumor_detected']:
            return "Patient MRI scan shows no evidence of tumor."

        bbox = features.get('bounding_box')
        bbox_str = f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})" if bbox else "N/A"

        report = (
            "MRI Brain Tumor Analysis Report:\n"
            f"- Tumor detected: Yes\n"
            f"- Confidence approx: {features['confidence']}%\n"
            f"- Approx Tumor area: {features['area_percent']}% of total brain slice\n"
            f"- Anatomical location: {features['location']}\n"
            f"- Morphological shape: {features['shape']}\n"
            f"- Bounding box (pixel coordinates): {bbox_str}\n\n"
            "Write a **concise clinical report** based only on the above MRI features. "
            "Use a medically appropriate tone, suitable for both clinicians and patients. "
            "Do NOT invent case studies, logic puzzles, or hypothetical patients. "
            "Do NOT add general medical advice or questions for the user. "
            "This is a one-way factual report with no conversation. "
            "Focus on describing the tumor’s characteristics, clinical implications, and follow-up recommendations, "
            "especially based on location, size, and shape. "
            "Avoid excessive jargon. Do NOT add unrelated content."
        )


        return report



    def call_llm_api(self, prompt_text: str) -> str:
        api_url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        json_data = {
            "model": "phi",
            "prompt": prompt_text,
            "max_tokens": 300,
            "temperature": 0.7,
            "stream": False
        }

        try:
            response = requests.post(api_url, headers=headers, json=json_data, timeout=1000)
            response.raise_for_status()

            # In case of multiple lines, take only the first JSON object
            first_line = response.text.strip().splitlines()[0]
            data = json.loads(first_line)

            return data.get("response", "").strip()
        except Exception as e:
            return f"Error generating clinical report: {str(e)}"
        
    # def extract_tumor_features(mask, image_shape):
    #     # Example feature extraction: area, bounding box, etc.
    #     tumor_detected = mask.sum() > 0
    #     area = int(mask.sum())
    #     location = "Unknown"

    #     # Find bounding box
    #     coords = np.column_stack(np.where(mask > 0))
    #     if coords.size > 0:
    #         y_min, x_min = coords.min(axis=0)
    #         y_max, x_max = coords.max(axis=0)
    #         bounding_box = (x_min, y_min, x_max, y_max)
    #         # Simple heuristic: left/right/center based on bbox center relative to width
    #         center_x = (x_min + x_max) // 2
    #         width = image_shape[1]
    #         if center_x < width / 3:
    #             location = "Left brain lobe"
    #         elif center_x > 2 * width / 3:
    #             location = "Right brain lobe"
    #         else:
    #             location = "Central brain lobe"
    #     else:
    #         bounding_box = None

    #     shape = "Irregular"  # You could add shape analysis if needed

    #     return {
    #         "tumor_detected": tumor_detected,
    #         "area": area,
    #         "location": location,
    #         "shape": shape,
    #         "bounding_box": bounding_box,
    #     }



    # def extract_tumor_features(self, mask):
    #     labeled = label(mask)
    #     regions = regionprops(labeled)
    #     if not regions:
    #         return {"present": False}

    #     tumor = max(regions, key=lambda r: r.area)
    #     features = {
    #         "present": True,
    #         "area": tumor.area,
    #         "centroid": tumor.centroid,
    #         "bounding_box": tumor.bbox,
    #         "shape": "elongated" if tumor.major_axis_length > 1.5 * tumor.minor_axis_length else "rounded"
    #     }
    #     return features

    # def generate_clinical_report(self, features):
    #     if not features["present"]:
    #         return "No tumor was detected in the MRI scan."

    #     y, x = features["centroid"]
    #     report = (
    #         f"A tumor has been detected in the brain MRI.\n"
    #         f"- Approximate area: {features['area']} pixels\n"
    #         f"- Location (centroid): ({int(x)}, {int(y)})\n"
    #         f"- Shape characteristic: {features['shape']}\n"
    #         f"- Bounding box: {features['bounding_box']}\n"
    #         f"Follow-up with a specialist is recommended for detailed analysis."
    #     )
    #     return report

    
    
