import torch
from flask import Flask, render_template, request, jsonify
from torchvision.utils import save_image
from model import Generator, load_model 

app = Flask(__name__)
model_path = 'generator.pth'  

generator = load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-image', methods=['POST'])
def generate_image():
    # Generate an image
    z = torch.randn(1, 100)  
    generated_image = generator(z)
    generated_image = generated_image.view(1, 1, 28, 28)  

    # Save the generated image
    save_image(generated_image, 'static/images/generated.png')

    return jsonify({'result': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
