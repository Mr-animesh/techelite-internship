
#!pip install torch torchvision pillow

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#image loading and preprocessing
def load_image(path, max_size=400, shape=None):
    image = Image.open(path).convert('RGB')
    
    # Resize
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])

    image = in_transform(image).unsqueeze(0)
    return image.to(device)

#show image helper
def im_convert(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image)


#load content and style image
content = load_image("virat-kohli.jpg").to(device)
style = load_image("artist_style.jpg", shape=content.shape[-2:]).to(device)

plt.imshow(im_convert(content))
plt.title("Content Image")
plt.show()

plt.imshow(im_convert(style))
plt.title("Style Image")
plt.show()


#load pretrained VGG
vgg = models.vgg19(pretrained=True).features.to(device).eval()

for param in vgg.parameters():
    param.requires_grad_(False)


#define layer names
def get_features(image, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',  # Content layer
        '28': 'conv5_1'
    }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


#gram matrix for style
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


#extract features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}


#create target image
target = content.clone().requires_grad_(True).to(device)

#define optimizer
optimizer = optim.Adam([target], lr=0.003)


#style transfer loop
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.75,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2
}

content_weight = 1e4 #tweak these fields to get less or more stylized image
style_weight = 1e2

steps = 2000

for step in range(1, steps+1):
    target_features = get_features(target, vgg)
    
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (target_feature.shape[1] ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"Step {step}, Total loss: {total_loss.item():.4f}")

#display final image
final_img = im_convert(target)
plt.imshow(final_img)
plt.title("Stylized Image")
plt.axis('off')
plt.show()

