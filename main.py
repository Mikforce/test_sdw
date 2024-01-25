from flask import Flask, render_template, request, send_file
import torch
from diffusers import StableDiffusionPipeline


model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        negative_prompt = request.form['negative_prompt']

        prompt = f"{prompt}, {negative_prompt}"

        image = pipe(prompt).images[0]

        image.save("output.png")

        return send_file('output.png', as_attachment=True)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
