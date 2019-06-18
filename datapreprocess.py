import numpy as np 
import mat4py 
import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader
import os 
from PIL import Image,ImageDraw
from skimage import io, transform
import sys 
from torchvision import transforms, utils
import copy 
import torch.nn.functional as F


# note that the folder is organized as 
# ../data/
LABEL_PATH = "../data/devkit/cars_train_annos.mat"
SAVE_CSV_PATH = "../data/cars_annos.csv"
ROOT_FOLDER = "../data/cars_train"

def create_labels_csv():
    dev = mat4py.loadmat(LABEL_PATH)
    dev = dev['annotations']
    df = pd.DataFrame.from_dict(dev)
    # get all classes of the cars 
    #Note that all numeric labels are decreased by one for training 
    #classes = np.array(df['class'])
    #classes = classes - 1 
    df.to_csv(SAVE_CSV_PATH)


def read_labels_csv_to_pddf(SAVE_CSV_PATH):
    df = pd.read_csv(SAVE_CSV_PATH)
    return df 


def convert_grey_to_RGB(img):
    if len(img.shape) == 3:
        #already in RGB 
        return img 
    assert len(img.shape) == 2, "not a greyscale img"
    dim = np.zeros((img.shape))
    RGBimg = np.stack((img, img, img), axis=2)
    return RGBimg 
    

### use transfer learning 
class CarDataset(Dataset):
    """ The car dataset uses the dataframe from the csv file to inrepret the pictures and classes 
    """
    def __init__(self, root_dir, label_df, transforms = None):
        # the labels is a simple python list that contains all the lables 
        # for the input dataset 
        self.root_dir = root_dir        
        self.transform = None 
        self.label_df = label_df
        if transforms:
            self.transforms = transforms
    
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        # returns a particular training example 
        # get the image name by idx --> rely on the label_df to indexing and get the class label 
        row = self.label_df.iloc[idx]
        img_name = row['fname']
        
        # create the image path 
        img_path = os.path.join(self.root_dir, img_name)
        img = io.imread(img_path)
        
        # if img is in grey-scale, convert it back to RGB 
        img = convert_grey_to_RGB(img)
        
        bbox = [row['bbox_x1'].tolist(), row['bbox_x2'].tolist(), row['bbox_y1'].tolist(), row['bbox_y2'].tolist()] 
        # convert to np array 
        bbox = np.array(bbox)
        label = np.array(row['class'].tolist())
        sample = {'image': img, 'label': label, 'bbox':bbox}
        if self.transforms:
            sample = self.transforms(sample)
        return sample 
    

    
def generate_small_cardataset(num_classes):
    """ generate a car classification dataset with randomly selected num_classes 
    
        This is created to faciliate fast training and testing process for network architecture construction 
    """
    # define the transforms     
    #cardataset = CarDataset(ROOT_FOLDER,label_df, transforms= transforms.Compose([Rescale(256),ToTensor()])) 
    df = read_labels_csv_to_pddf(SAVE_CSV_PATH)
    total_num_classes = len(set(df['class'].tolist()))
    assert num_classes <= total_num_classes, "the input num class is larger than total number of classes"

    #sample num classes --> class number is not zero indexed 
    sampled_classes = np.random.choice([i for i in range(1,total_num_classes+1)],num_classes ,replace=False)
    print(sampled_classes)
    
    # relabel the training data and save the label map 
    label_map = {} 
    for i in range(len(sampled_classes)):
        label_map[sampled_classes[i]] = i 
    print(label_map)
    
    #filter the label df 
    sampled_df = df[df['class'].isin(sampled_classes)]
    
    for i in range(len(sampled_df)):
        class_number = sampled_df.iloc[i]['class']
        mapped_number = label_map[class_number]        
        sampled_df.iloc[i, sampled_df.columns.get_loc('class')]  = mapped_number
    sampled_df.reset_index(drop=True, inplace=True)

    print(str(len(sampled_df)) + " samples are selected.")

    #construct a dataset based on the selected samples 
    cardataset = CarDataset(ROOT_FOLDER, sampled_df, transforms= transforms.Compose([Rescale(256),ToTensor()])) 
    return cardataset 
    
    
    
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']

        h, w = image.shape[:2]
        
        # this function applies uniform transform--> we want all input images to be sqaure 
        # need a way to convert it back to the original scale --> could be implemented with some meta background knowledge 
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * w / h, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * h / w
        else:
            new_h, new_w = self.output_size
        #new_h, new_w = int(new_h), int(new_w)
        
        new_h, new_w = int(self.output_size), int(self.output_size)
        
        #print(new_h,new_w)
        # the resize function would normalize the image to 0-1 scale -- we want to convert it back for display purposes 
        img = transform.resize(image, (new_h, new_w))
        img = 255 * img
        # Convert to integer data type pixels.
        img = img.astype(np.uint8)
       
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        
        # figure out the scale  x=10
        
        bbox = [bbox[0]/w * new_w, bbox[1]/w * new_w, bbox[2]/h * new_h, bbox[3]/h * new_h]
        bbox = np.array(bbox)
        sample['image'] = img
        sample['bbox'] = bbox 
        
        return sample



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bbox, label = sample['image'], sample['bbox'], sample['label']
        #print(image.shape)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        bbox = torch.from_numpy(bbox).type('torch.FloatTensor')
        # type for classification target is Long 
        label = torch.from_numpy(label).type('torch.LongTensor')
        
        #convert to FloatTensor Type to be compatible with the model parameters
        image = image.type('torch.FloatTensor')
        bbox = bbox.type('torch.FloatTensor')
        label = label.type('torch.FloatTensor')
        
        sample['image'] = image
        sample['bbox'] = bbox 
        sample['label'] = label
        return sample
    


def draw_bounding_box(image, bbox, save_path):
    """ The input sample should already be preprocessed 
        and we assume the image is a tensor 
    
    """
    
    # if the image is a Float tensor type..  
    
    img = copy.deepcopy(image)
    img = img.numpy()
    img = img.transpose((1,2,0))
    image = Image.fromarray(img)
    bbox = copy.copy(bbox)
    bbox = bbox.cpu().detach().numpy()
    draw = ImageDraw.Draw(image)
    draw.rectangle([int(bbox[0]),int(bbox[2]), int(bbox[1]), int(bbox[3]) ])
    del draw 
    image.save( save_path, "PNG")

    
    
def main():
    create_labels_csv()
    label_df = read_labels_csv_to_pddf(SAVE_CSV_PATH)
    
    # define the transforms     
    cardataset = CarDataset(ROOT_FOLDER,label_df, transforms= transforms.Compose([Rescale(256),ToTensor()])) 
    
    sample  = cardataset[1]
    draw_bounding_box(sample)
    img = sample['image'] 
    img = np.array(img)





if __name__ == "__main__":
    generate_small_cardataset(10)


