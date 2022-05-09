import argparse, torch, sys, os, numpy as np, torchvision.transforms as transforms, torch.optim as optim, torch.nn as nn, wandb
from torchvision import datasets
from torch.utils.data.dataloader import DataLoader
from model import DarkNet53
from torchsummary import summary

def parse_args():
    parser = argparse.ArgumentParser(description="DarkNet53_Cat&Dog")
    parser.add_argument("--mode", dest="mode", default=None, type=str)
    parser.add_argument("--output_dir", dest="output_dir", default="output", type=str)
    parser.add_argument("--pretrain", dest="pretrain", default=0, type=int)
    args = parser.parse_args()
    return args

def start():
    print("#"*35)
    print("\ttorch version: ", torch.__version__)
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    if torch.cuda.is_available():
        print("\tcuda is available!")
        device = torch.device("cuda")
    else:
        ("\tcuda is unavailable!")
        device = torch.device("cpu")
    print("#"*35)
    return device
    
def get_dataset():
    train_dir = os.path.join("data", "train")
    eval_dir = os.path.join("data", "eval")
    test_dir = os.path.join("data", "test")
    
    my_transform = transforms.Compose([transforms.Resize((256, 256)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])
                                       ])
    
    train_dataset = datasets.ImageFolder(train_dir, my_transform)
    eval_dataset = datasets.ImageFolder(eval_dir, my_transform)
    test_dataset = datasets.ImageFolder(test_dir, my_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=0, pin_memory=True, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, pin_memory=True, drop_last=True, shuffle=True)
    
    return train_loader, eval_loader, test_loader

def get_pretrain(model):
    if args.pretrain == 1:
        print("DarkNet53 transfer learning mode!")
        pretrain = torch.load("model_best.pth.tar", map_location=device)
        pretrained_state_dict = pretrain['state_dict']
        model_state_dict = model.state_dict()                 
        for key, value in pretrained_state_dict.items():
            # skip fully-connected layer in pretrained weights.
            # because the ouput channel of fc layer is dependent on number of classes.
            if key == 'fc.weight' or key == 'fc.bias':
                continue
            else:
                model_state_dict[key] = value
        model.load_state_dict(model_state_dict)
    
    else:
        print("DarkNet53 normal train mode!")
        
    return model

        

def train(train_loader, eval_loader):
    wandb.init(project="New DarkNet53", entity="openjr")
    model = DarkNet53()
    model = get_pretrain(model)
    model.to(device)
    model.train()
    
    # optimizer & scheduler & loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.1) #momentum=0.9,
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.BCELoss()
    
    
    for epoch in range(epochs):
        # Train
        
        train_acc = 0
        for i, (train_img, train_gt) in enumerate(train_loader):
            train_img, train_gt = train_img.to(device), train_gt.view(-1, 1).float().to(device)
            train_out = model(train_img)
            
            optimizer.zero_grad()
            train_loss_val = criterion(train_out, train_gt)
            train_loss_val.backward()
            optimizer.step()
            
            train_out = torch.round(train_out)
            train_acc += (train_out == train_gt).sum().item()
            
            train_acc_log = train_acc/(train_img.shape[0]*(i+1))*100
            wandb.log({"train_loss":train_loss_val.item(), "train_acc":train_acc_log})
            print("epoch: {}/{}, i: {}/{}, train loss: {}, train acc: {}".format(epoch+1, epochs, i+1, len(train_loader)+1,  round(train_loss_val.item(), 4), round(train_acc_log, 2)))
            
            # Evaluation
            model.eval()
            eval_acc = 0
            eval_loss = 0
            with torch.no_grad():
                for _, (eval_img, eval_gt) in enumerate(eval_loader):
                    eval_img, eval_gt = eval_img.to(device), eval_gt.view(-1, 1).float().to(device)
                    eval_out = model(eval_img)
                    
                    eval_loss_val = criterion(eval_out, eval_gt)
                    eval_out = torch.round(eval_out)
                    eval_acc += (eval_out == eval_gt).sum().item()
                    eval_loss += eval_loss_val.item()
            
            eval_acc_log = eval_acc / len(eval_loader.dataset) * 100
            eval_loss_log = eval_loss / len(eval_loader)
            wandb.log({"eval_loss":eval_loss_log, "eval_acc":eval_acc_log})
            print("epoch: {}/{}, i: {}/{}, eval loss: {}, eval acc: {}".format(epoch+1, epochs, i+1, len(train_loader)+1,  round(eval_loss_log, 4), round(eval_acc_log, 2)))
            model.train()
            
        scheduler.step()
          
    torch.save(model.state_dict(), args.output_dir + "/DarkNet53.pt")
            
            
def main():
    train_loader, eval_loader, test_loader = get_dataset()
    
    if args.mode == "train":
        train(train_loader, eval_loader)
    
if __name__ == "__main__":
    args = parse_args()
    device = start()
    epochs, batch_size = 1, 32
    main()