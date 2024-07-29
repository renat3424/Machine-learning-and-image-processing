import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights
import torch.optim as optim
from torchvision.models import resnet50
from torchvision.transforms import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchsummary import summary
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("here")
model=vgg19(weights=VGG19_Weights.DEFAULT).to(device=device)

content_layers=["features.31"]
style_layers=["features.1", "features.6", "features.11", "features.20", "features.29"]



model.eval()
model2=create_feature_extractor(model, content_layers+style_layers)

def content_loss(real, styled):

    return torch.mean(torch.square(real-styled), dim=None)

def gram_matrix(input_matrix):
    channels=input_matrix.shape[1]
    input_matrix=input_matrix.reshape([1, channels, -1])
    input_matrix=input_matrix.squeeze(0)

    return torch.matmul(input_matrix, torch.transpose(input_matrix, 0, 1))/input_matrix.shape[1]

def get_style_loss(base, gram_target):
    gram_base=gram_matrix(base)

    return torch.mean(torch.square(gram_base-gram_target), dim=None)

def compute_loss(model, loss_weights, init_image, style_features, content_features):
    style_weight, content_weight=loss_weights
    style_score=0
    content_score=0
    image_outputs=model(init_image)

    for i, (key, value) in enumerate(image_outputs.items()):
        if key!="features.31":

            style_score += get_style_loss(image_outputs[key], style_features[key])/5
        else:

            content_score += content_loss(content_features, image_outputs[key])
    print(style_score.item(), content_score.item())
    total_loss=style_weight*style_score+content_weight*content_score
    return total_loss, style_score, content_score


def transform_image(img_path):
    img=Image.open(img_path)
    img=np.asarray(img, dtype=np.float32)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    if img.shape==(224, 224, 3):


        transform=transforms.Compose([transforms.ToTensor()])
        img=transform(img)
    else:

        transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(400),
             transforms.CenterCrop(400)])
        img = transform(img)
    return img





real_image=transform_image("фото для резюме.jpg").to(device=device)
style_image=transform_image("picasso.jpg").to(device=device)
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)
style_features=model2(style_image)
style_features={key: gram_matrix(style_features[key]) for key in style_features.keys()}
content_features=model2(real_image)["features.31"]
num_iterations=10000
content_weight=1000
style_weight=1

image=torch.clone(real_image)
image.requires_grad=True
optimizer=optim.Adam([image], lr=8, eps=0.1, betas=(0.99, 0.999))
iter_count=1
best_loss, best_img=np.inf, None
loss_weights=(style_weight, content_weight)
cfg={"model": model2, "loss_weights": loss_weights, "init_image": image, "style_features": style_features, "content_features": content_features}
for i in range(num_iterations):

    optimizer.zero_grad()
    total, loss1, loss2 = compute_loss(**cfg)

    total.backward(retain_graph=True)
    optimizer.step()

    if total.item() < best_loss:
        best_loss=total.item()
        best_img= image.detach().permute(1, 2 ,0)
        print(f"Iteration: {i}")
    with torch.no_grad():
        image.clamp_(0, 255)



best_img=best_img.to(device="cpu")


f, axarr = plt.subplots(1, 2)
axarr[0].imshow(best_img/255)

axarr[1].imshow(real_image.to(device="cpu").permute(1, 2 ,0)/255)
print(best_img.numpy().shape)
Image.fromarray(np.uint8(best_img.numpy())).save("new_img.jpg")

plt.show()