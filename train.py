import os
import platform
import argparse
import torch
from torch import nn
from torchvision import transforms
from timeit import default_timer as timer
import data_setup, engine, model_builder, utils

MANUAL_SEED = 42
NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup hyperparameters (can use python's argparse to parse from command line)
parser = argparse.ArgumentParser(
    prog='Image recognition with TinyVGG',
    description='Determine whether an image is of a pizza, steak or sushi',
    epilog='Have fun!')
parser.add_argument('--num_epochs',    default=NUM_EPOCHS,    type=int)
parser.add_argument('--batch_size',    default=BATCH_SIZE,    type=int)
parser.add_argument('--hidden_units',  default=HIDDEN_UNITS,  type=int)
parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float)
args = parser.parse_args()
print(f"Creating mode with parameters:\n"
      f"\tnum_epochs={args.num_epochs}\n"
      f"\tbatch_size={args.batch_size}\n"
      f"\thidden_units={args.hidden_units}\n"
      f"\tlearning_rate={args.learning_rate}\n")

if platform.system() == "Darwin":  # macOS
    """ DataLoader with num_workers > 0 can be significantly slower on macOS due to the way multiprocessing and file I/O
    are handled by the OS. This is not a problem on Linux, but on macOS, using multiple workers often leads to overhead
    that outweighs the benefits. Set num_workers=0 in your create_dataloaders call (as you already do in train.py). This
    will make data loading single-threaded, which is usually faster and more stable on macOS."""
    NUM_WORKERS = 0
else:
    NUM_WORKERS = os.cpu_count()

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup device agnostic code
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main():
    # Create transforms (can move to transforms.py)
    data_transform = transforms.Compose([
      transforms.Resize((64, 64)),
      transforms.ToTensor()
    ])

    # create Dataloader and get classnames
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=args.batch_size,
        num_workers=0
    )

    # Set random seeds
    torch.manual_seed(MANUAL_SEED)
    torch.cuda.manual_seed(MANUAL_SEED)
    torch.mps.manual_seed(MANUAL_SEED)

    # Recreate an instance of TinyVGG
    model_0 = model_builder.TinyVGG(input_shape=3, # number of color channels (3 for RGB)
                      hidden_units=args.hidden_units,
                      output_shape=len(class_names)).to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=args.learning_rate)

    # Start the timer
    start_time = timer()

    # Train model_0
    model_0_results = engine.train(model=model_0,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=args.num_epochs,
                            device=device)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

    # Save the model
    utils.save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth",
               model_args={
                     "input_shape": 3,
                     "hidden_units": args.hidden_units,
                     "output_shape": len(class_names)
               })

if __name__ == "__main__":
    main()