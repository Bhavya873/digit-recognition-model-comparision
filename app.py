import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image as PILImage
import numpy as np
from cnn import VGG11
from perceptron import load_perceptron

# 1. Load device & models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = VGG11().to(device)
vgg.load_state_dict(torch.load("vgg11_mnist.pth", map_location=device, weights_only=False))
vgg.eval()
perc = load_perceptron("perceptron_ova_mnist.pth", device)

# 2. Transforms for each model
transform_vgg = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_perc = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

MODEL_NAMES = ["VGG-11", "Perceptron"]

# 3. Safely extract the drawn image array, defaulting to blank white
def unwrap_editor(editor_output):
    """
    Always extract a valid NumPy array from the editor output,
    defaulting to a blank canvas if nothing is present.
    """
    # 1) If Gradio returns a dict (Sketchpad/ImageEditor), check each key
    if isinstance(editor_output, dict):
        if "composite" in editor_output and editor_output["composite"] is not None:
            arr = editor_output["composite"]
        elif "image" in editor_output and editor_output["image"] is not None:
            arr = editor_output["image"]
        elif "background" in editor_output and editor_output["background"] is not None:
            arr = editor_output["background"]
        else:
            # nothing drawn → blank white canvas
            return 255 * np.ones((600, 600), dtype=np.uint8)
    else:
        arr = editor_output

    # 2) If it’s still None, return blank canvas
    if arr is None:
        return 255 * np.ones((600, 600), dtype=np.uint8)

    # 3) If it’s a PIL Image, convert to grayscale NumPy array
    if isinstance(arr, PILImage.Image):
        arr = np.array(arr.convert("L"))

    # 4) Otherwise assume it’s already a NumPy array
    return arr


# 4. Run the chosen model on the array
def recognize_with_choice(arr, model_name):
    # Blank canvas → zero scores
    if np.all(arr == 255):
        return {str(i): 0.0 for i in range(10)}

    img = PILImage.fromarray(arr).convert("L")
    img = PILImage.eval(img, lambda x: 255 - x)  # invert

    if model_name == "VGG-11":
        x = transform_vgg(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(vgg(x), dim=1)[0]
        return {str(i): float(probs[i]) for i in range(10)}
    else:
        x = transform_perc(img).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = perc(x)
            pred = int(scores.argmax())
        return {str(i): (1.0 if i == pred else 0.0) for i in range(10)}

# 5. Format results into HTML table
def recognize_and_format(editor_output, model_name):
    arr = unwrap_editor(editor_output)
    scores = recognize_with_choice(arr, model_name)
    pred = max(scores, key=scores.get)

    rows = ""
    for i in range(10):
        highlight = " class='pred-row'" if str(i) == pred else ""
        rows += f"<tr{highlight}><td>{i}</td><td>{scores[str(i)]:.3f}</td></tr>"

    return (
        f"<div style='text-align:center;font-size:1.4em;"
        f"margin-bottom:12px;color:#24517a;font-weight:600;'>"
        f"Prediction: {pred}</div>"
        "<table id='score-table'>"
          "<thead><tr><th>Digit</th><th>Score</th></tr></thead>"
          f"<tbody>{rows}</tbody>"
        "</table>"
    )

# 6. Function to clear the canvas
def clear_canvas():
    return 255 * np.ones((600, 600), dtype=np.uint8)

# 7. Custom CSS for layout and styling
css = """
body, html {margin:0;padding:0;height:100%;width:100%;
            background: linear-gradient(120deg,#f7fafc,#e3eefa);}
.gradio-container {display:flex!important;flex-direction:column;
                   justify-content:center;align-items:center;
                   height:100%;width:100%;}
#main-row {display:flex;width:100%;padding:20px;
           justify-content:space-around;align-items:center;}
#controls-col {display:flex;flex-direction:column;
               align-items:center;gap:20px;}
#canvas {border:2px solid #7db7e8;border-radius:8px;
         box-shadow:0 0 12px rgba(200,230,250,0.4);}
#model-radio {display:flex!important;gap:40px!important;
               justify-content:center!important;}
#model-radio label {font-size:1.1em!important;font-weight:600!important;
                    color:#24517a!important;}
#clear-btn {background:#ff4c4c;color:#fff;border:none;
            min-width:120px;height:44px;border-radius:6px;
            font-size:1em;}
#clear-btn:hover {background:#c62828;}
#score-table {margin-top:20px;border-collapse:collapse;
              width:100%;max-width:400px;text-align:center;
              background:#fff;border-radius:8px;overflow:hidden;
              box-shadow:0 2px 8px rgba(0,0,0,0.1);}
#score-table th,#score-table td {border:1px solid #ddd;
                                padding:10px;color:#24517a;}
#score-table th {background:#f2f8fc;font-weight:600;}
#score-table tr:nth-child(even){background:#f9f9f9;}
#score-table tr.pred-row{background:#ffff99;font-weight:600;}
#score-table tr.pred-row td{color:#333300;}
"""

# 8. Build and launch the Gradio interface
with gr.Blocks(css=css) as demo:
    gr.Markdown(
        "<h2 style='text-align:center;color:#24517a;margin-bottom:10px;'>"
        "Digit Recognizer</h2>"
    )
    gr.Markdown(
        "<p style='text-align:center;color:#4a607a;margin-bottom:30px;'>"
        "Draw a digit to see live predictions</p>"
    )

    with gr.Row(elem_id="main-row"):
        canvas = gr.Sketchpad(
            type="numpy",
            height=600,
            width=600,
            label="Draw Digit",
            elem_id="canvas"
        )
        with gr.Column(elem_id="controls-col"):
            model_radio = gr.Radio(
                choices=MODEL_NAMES,
                value="VGG-11",
                show_label=False,
                interactive=True,
                elem_id="model-radio"
            )
            clear_btn = gr.Button("Clear", elem_id="clear-btn")
            # Initialize the score table with zeros
            initial_rows = "".join(f"<tr><td>{i}</td><td>0.000</td></tr>" for i in range(10))
            score_out = gr.HTML(
                "<table id='score-table'><thead><tr><th>Digit</th><th>Score</th></tr></thead>"
                f"<tbody>{initial_rows}</tbody></table>",
                elem_id="score-table"
            )

    # Live updates as you draw or switch models
    canvas.change(
        recognize_and_format,
        inputs=[canvas, model_radio],
        outputs=score_out,
        queue=False
    )
    model_radio.change(
        recognize_and_format,
        inputs=[canvas, model_radio],
        outputs=score_out,
        queue=False
    )
    clear_btn.click(
        clear_canvas,
        outputs=canvas,
        queue=False
    )

if __name__ == "__main__":
    demo.launch()
