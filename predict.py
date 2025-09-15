import argparse
import torch
import utils
import torchvision
from torchvision import transforms

parser = argparse.ArgumentParser(
    prog='Image recognition with TinyVGG',
    description='Determine whether an image is of a pizza, steak or sushi',
    epilog='Have fun!')
parser.add_argument('--file_name', type=str)
args = parser.parse_args()

print(f"[INFO] Predict the classname for file: {args.file_name}")

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Convert image into the same format which the model was trained on
# 1. Convert to tensor (torch.float32)
# 2. Shape 64x64
# 3. Put it on a right device

custom_image_uint = torchvision.io.read_image(args.file_name).type(torch.float32)

# plt.imshow(custom_image_uint.permute(1, 2, 0))

print(f'custom_image_uint.shape={custom_image_uint.shape}')
custom_image_transform = transforms.Compose([transforms.Resize((64, 64))])
custom_image_uint = custom_image_transform(custom_image_uint)
print(f'custom_image_uint.shape={custom_image_uint.shape}')

model = utils.load_model("models", "05_going_modular_tingvgg_model.pth")
model.to(device) # Проверить будет ли работать если модель обучалась на одном устройстве, а инференс делать на другом
model.eval()

with torch.inference_mode():
    # Деление на 255 - я не понял почему именно на 255. В исходных данных
    # из набора pizza_steak_sushi все картинки когда уложены в тенсор
    # имеют значения от 0 до 1 (я думаю, благодаря DataLoader'у который
    # сделал это автоматически), но когда мы упаковали эту тестовую картинку
    # (а мы сделили это вручную) все значения тенсора оказались целыми значениями от 0 до 255
    # которые мы преобразовали в float32, теперь чтобы привести их к вижу исходной
    # коллекции поделим каждое значение тенсора ра 255
    custom_image_float = custom_image_uint.type(torch.float32) / 255.
    # Несмотря на то что мы поделили тенсор на 255, картинка остается без измнений:
    # plt.imshow(custom_image_float.permute(1, 2, 0)) # permute используется для CHW -> HWC
    # У нас одна картинка, а модель принимает их батчами, обернем эту картинку
    # в дополнительный массив [x,y,z] -> [1,x,y,z], получится батч из одной картинки
    custom_image_batch = custom_image_float.unsqueeze(dim=0)
    custom_image_batch = custom_image_batch.to(device)
    prediction_logits = model(custom_image_batch.to(device))
    # print(f'image_pred={torch.argmax(image_pred)}')
    prediction = ["sushi", "steak", "pizza"][torch.argmax(prediction_logits).item()]
    # полезное свойство, хорошо натренированная модель будет выдавать значения близкие
    # к единице, потому как модель уверенна в своем выборе. Плохая модель будет иметь
    # в probability значения, которые похожи на догадки. Например, если у нас три
    # возможных варианта ответа и модель выдает ответ с вероятностью 0.345, то вероятно
    # то вероятно, что модель выбрала на угад один из трех ответов, где вероятность
    # была на сотую профента больше.
    probability = torch.softmax(prediction_logits, dim=1).max().cpu()
    print(f"Prediction: {prediction}, Probability: {probability:.3f}")