import gradio as gr
import numpy as np
from PIL import Image
from typing import Tuple, List, Union
import os
from model import SIRENa
from config import Config


def process_image(image_path: Union[str, np.ndarray]):
    """
    Mock model processing function. Replace this with your actual model.

    Args:
        image_path: Path to input image or numpy array from Gradio

    Returns:
        Tuple containing dictionary of label->subtitle and list of processed images
    """
    global model
    outputs = model(image_path)
    config = Config()

    # Process images for first n elements
    processed_images = [
        np.array(Image.open(f"results/{i}.png")) for i in range(config.seg_n_masks)
    ]

    return outputs, processed_images


def create_interface():
    """Creates and launches the Gradio interface"""
    css = """
#warning {background-color: #FFCCCB}
.feedback textarea {font-size: 24px !important}

/* Style for logo container */
.logo-container {
    text-align: left;
    padding: 20px 0;
}

.logo-container img {
    max-height: 100px;  /* Adjust this value as needed */
    width: auto;
}
"""

    theme = gr.themes.Default(
        # primary_hue="red",
    )

    with gr.Blocks(css=css, theme=theme) as demo:
        # Logo section
        with gr.Row(elem_classes="logo-container"):
            gr.Image(
                "logo.png", show_label=False, container=False
            )  # Replace with your logo path

        gr.Markdown("# SIRENα")
        gr.Markdown("## Team 27")

        with gr.Row():
            with gr.Column():
                # Input components
                input_image = gr.Image(label="Input Image", type="filepath")
                submit_btn = gr.Button("Analyze Image")

            with gr.Column():
                detection_containers = []
                # Pre-create 10 containers (or adjust based on expected max detections)
                for i in range(10):
                    with gr.Accordion(
                        f"Detection {i+1}", visible=False, open=False
                    ) as acc:
                        section_image = gr.Image(
                            show_label=False, height=200, visible=False
                        )
                        section_text = gr.Markdown(visible=False)
                        detection_containers.append(
                            {
                                "accordion": acc,
                                "image": section_image,
                                "text": section_text,
                            }
                        )

        def process_and_format(image):
            outputs_dict, images = process_image(image)

            # Prepare updates for all containers
            container_updates = []
            for i in range(len(detection_containers)):
                if i < len(outputs_dict.keys()):
                    label = list(outputs_dict.keys())[i]
                    subtitle = outputs_dict[label]
                    has_image = i < len(images)

                    # Updates for this container
                    container_updates.extend(
                        [
                            gr.update(
                                visible=True, label=label.capitalize(), open=(i == 0)
                            ),  # Accordion
                            gr.update(
                                visible=has_image,
                                value=images[i] if has_image else None,
                            ),  # Image
                            gr.update(visible=True, value=subtitle.capitalize()),  # Text
                        ]
                    )
                else:
                    # Hide unused containers
                    container_updates.extend(
                        [
                            gr.update(visible=False),  # Accordion
                            gr.update(visible=False),  # Image
                            gr.update(visible=False),  # Text
                        ]
                    )

            return container_updates

        # Flatten the outputs list
        outputs = []
        for container in detection_containers:
            outputs.extend(
                [container["accordion"], container["image"], container["text"]]
            )

        # Set up the main event handler
        submit_btn.click(fn=process_and_format, inputs=[input_image], outputs=outputs)

    return demo


if __name__ == "__main__":
    global model
    model = SIRENa()

    demo = create_interface()
    demo.launch()
