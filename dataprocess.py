from torchvision import transforms


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
crop_size = (224, 224)

Transform_train = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.RandomCrop((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)
                                      ])

Transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)
                                     ])

