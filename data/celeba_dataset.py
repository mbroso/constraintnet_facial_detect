"""Class for loading CelebA dataset.
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, SequentialSampler
import torchvision
from torchvision import transforms
import pandas as pd
from PIL import Image
from skimage import io, transform
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import re
from abc import ABC, abstractmethod
import functools
import random
import time

from data.celeba_plugins.SeqSampler import SeqSampler
from data.celeba_plugins.constr_para_generator_bb import opts2face_bb_rand
from data.celeba_plugins.constr_para_generator_polytope import opts2lm_polytope_rand
from data.celeba_plugins.constr_para_generator_circle_sector import opts2lm_circle_sector_rand
from data.celeba_plugins.lm_ordering import opts2lm_ordering

def opts2celeba_dataset(opts):
    """Creates CelebaDataset object by calling its constructor with options
    from opts.

    Args:
        opts (obj): Namespace object with options.

    Returns:
        celeba_dataset (obj): Instantiated CelebaDataset object.
    """
    #dictionary to map opts2constr_para_generator string to object
    opts2constr_para_generator_dict = {
            'None': None,
            }
    
    if hasattr(opts, 'opts2constr_para_generator'): #opts.model_module == 'constraintnet':
        if opts.opts2constr_para_generator == 'opts2face_bb_rand':
            opts2constr_para_generator_dict['opts2face_bb_rand'] =  opts2face_bb_rand(opts)
        if opts.opts2constr_para_generator == 'opts2lm_polytope_rand':
            opts2constr_para_generator_dict['opts2lm_polytope_rand'] =  opts2lm_polytope_rand(opts)
        if opts.opts2constr_para_generator == 'opts2lm_circle_sector_rand':
            opts2constr_para_generator_dict['opts2lm_circle_sector_rand'] = opts2lm_circle_sector_rand(opts)
    constr_para_generator = None
    if hasattr(opts, 'opts2constr_para_generator'): #opts.model_module == 'constraintnet':
        constr_para_generator = \
                opts2constr_para_generator_dict[opts.opts2constr_para_generator]

    opts2y_generator_dict = {
            'None': None,
            'opts2lm_ordering': opts2lm_ordering(opts)
            }

    y_generator = opts2y_generator_dict[opts.opts2y_generator] 


    return CelebaDataset(Path(opts.imgs_dir), Path(opts.lms_file), 
            opts.preprocess, opts.preprocess_lm_keys, opts.rescale_output_size, 
            opts.randcovercrop_output_size, opts.randcovercrop_lms_covered, 
            opts.randcovercrop_padding, opts.randcovercrop_no_rand, 
            opts.randhorflip_p, opts.normalize_mean, opts.normalize_std,
            constr_para_generator, y_generator, opts.rotate_y, opts.sampler)


class CelebaDataset(Dataset):
    """This class implements the abstract base class Dataset for CelebA 
    dataset.
    """
    def __init__(self, imgs_dir, lms_file, preprocess=[], preprocess_lm_keys=[], 
            rescale_output_size=100, randcovercrop_output_size=100, 
            randcovercrop_lms_covered=['nose'], randcovercrop_padding=0. , 
            randcovercrop_no_rand = False, randhorflip_p = 0. , 
            normalize_mean = [0.485, 0.456, 0.406], 
            normalize_std = [0.229, 0.224, 0.225],
            constr_para_generator=None, y_generator=None, rotate_y=0., sampler='all'):
        """Initialization for accessing CelebA dataset.

        Args:
            imgs_dir (obj): Path (Path object of pathlib) to the images of the 
                dataset.
            lms_file (obj): Path (Path object of pathlib) to the txt file 
                containing the landmarks.
            preprocess (list): List of preprocessing steps.
            preprocess_lm_keys (list): List of landmark keys that should be in 
                the output sample. When no landmarks are specified all 
                landmarks are considered.
            rescale_output_size (int or list with two ints): Image size after 
                rescaling.
            randcovercrop_output_size (int or list with two ints): Image size 
                after random cover crop.
            randcovercrop_lms_covered (list): List of landmark keys that should 
                be covered by random cover crop.
            randcovercrop_padding (float or list of two floats): Padding as 
                fraction w.r.t. image edge. If one number is specified same 
                padding ratio is used for x and y dimension. With two numbers 
                padding rates for both dimensions can be specified separately. 
                Numbers must be between 0 and 1.
            randcovercrop_no_rand (boolean): No random sampling of crop 
                position.
            randhorflip_p (float): Probability for horizontal flip.
            normalize_mean (list): Mean value for each channel. Normalization 
                is computed according to input[channel] = (input[channel] - 
                mean[channel]) / std[channel]. Default values are for using 
                pretrained torch models.
            normalize_std (list): Standard deviations for each channel. 
                Normalization is computed according to input[channel] = 
                (input[channel] - mean[channel]) / std[channel]. Default values 
                are for using pretrained torch models.
            constr_para_generator (obj): If 'None' no constraint parameters are
                added. Otherwise this functor object creates constraint 
                parameters based on the preprocessed sample. The keyword 
                constr_para is added to the sample with an appropriate one 
                dimensional torch tensor created by this functor as value.
            y_generator (obj): If 'None' no y object is added to the sample and
                landmarks can be used. Otherwise the keyword y is added to the 
                preprocessed sample and the value is generated by this functor.
            rotate_matrix (list): List with 4 elements (r_11, r_12, r_21, r_22).
                These elements are the entries of a 2x2 rotation matrix. If this
                matrix is specified it is applied to y after y_generator was
                applied.

        """
        self.imgs_dir = imgs_dir
        self.lms_file = lms_file
        self.preprocess = preprocess
        self.preprocess_lm_keys = preprocess_lm_keys
        self.rescale_output_size = rescale_output_size
        self.randcovercrop_output_size = randcovercrop_output_size
        self.randcovercrop_lms_covered = randcovercrop_lms_covered
        self.randcovercrop_padding = randcovercrop_padding
        self.randcovercrop_no_rand = randcovercrop_no_rand 
        self.randhorflip_p = randhorflip_p
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.constr_para_generator = constr_para_generator
        self.y_generator = y_generator
        self.rotate_y = rotate_y

        self.lm_frame = pd.read_csv(lms_file, sep='\s+', header=1)
        self.all_lm_keys = self._get_all_lm_keys()
        self.lm_keys = self._get_lm_keys()
        
        
        #mapping from self.preprocess instantiation method
        self.preprocess_dict = {
                'rescale': self._preprocess_rescale,
                'randcovercrop': self._preprocess_randcovercrop,
                'randhorflip': self._preprocess_randhorflip,
                'totensor': self._preprocess_totensor,
                'normalize': self._preprocess_normalize,
                }
        #transformation function if no preprocessing is applied it is None
        self.transform = self._preprocess()



    def __getitem__(self, idx):
        """Implements sample access with bracket operator.

        If dataset is an instance of this class, access datasamples by typing
        dataset[idx]. The returned sample is in the following format 
        {'img': img, 'nose': np.array(nose_x, nose_y), 'lefteye': ...}.

        Args:
            idx (int): Index for accessing certain sample.

        Returns:
            sample (dict): Data sample with specified index. Sample is a 
                dictionary object with an 'img' and 5 landmark keys 'lefteye',
                'righteye', 'nose', 'leftmouth', 'rightmouth'. When 
                constr_para_generator and y_generator are specified the 
                keywords 'constr_para' and 'y' are created.
        """
        sample = {}

        img_name = self.lm_frame.iloc[idx].name
        img_path = self.imgs_dir / img_name
        img = io.imread(img_path)
        sample['img'] = img    
        for lm_key in self.lm_keys:
            xy = self.lm_frame.loc[
                    img_name,
                    [lm_key + '_x', lm_key+'_y']
                    ]
            xy = xy.values
            sample[lm_key] = xy
        
        if self.transform:
            sample = self.transform(sample)

        if not self.constr_para_generator is None:
            constr_para = self.constr_para_generator(sample)
            sample['constr_para'] = constr_para

        if not self.y_generator is None:
            y = self.y_generator(sample)
            sample['y'] = y

        if isinstance(self.rotate_y, list):
            if len(self.rotate_y) == 4:
                r_11 = self.rotate_y[0]
                r_12 = self.rotate_y[1]
                r_21 = self.rotate_y[2]
                r_22 = self.rotate_y[3]
                y_1 = sample['y'][0] 
                y_2 = sample['y'][1] 
                sample['y'][0] = y_1*r_11 + y_2*r_12
                sample['y'][1] = y_1*r_21 + y_2*r_22

        return sample
        
    def __len__(self):
        """Returns the number of samples in the dataset.

        Args:
            
        Returns:
            len (int): Number of samples in total.
        """
        return len(self.lm_frame)   

    def split(self):
        """Splits indices of the dataset into training, validation and test 
        part.

        Args:

        Returns:
            split (dict): Dictionary with indice lists for training, validation 
                and test set. The keys are train, valid and test respectively.
        """

        indices_train = list(range(162771))
        indices_valid = list(range(162771, 182638))
        indices_test = list(range(182638, self.__len__()))
        
        split = {
                'train': indices_train, 
                'valid': indices_valid,
                'test': indices_test
                }
        
        return split

    def get_sampler(self, indices):
        """Returns random sampler for specified indices.

        The sampler can be passed to the dataloader for sampling only from the
        subset given by indices.

        Args:
            indices (list): List with indices of subset. Elements must be 
                within [0, len(dataset)].

        Returns:
            sampler (obj): Torch.utils.data.SubsetRandomSampler for indices.
        """

        sampler = SubsetRandomSampler(indices)
        return sampler


    def get_fixed_sampler(self, indices):
        """Returns deterministic sampler for specified indices.

        The sampler can be passed to the dataloader for sampling only from the
        subset given by indices.

        Args:
            indices (list): List with indices of subset. Elements must be
                within [0, len(dataset)].

        Returns:
            sampler (obj): Torch.utils.data.SequentialSampler for indices.
        """
        sampler = SeqSampler(indices)

        return sampler


    def _get_all_lm_keys(self):
        """Access to all keys in the data sample with respect to landmarks.

        The keys related to landmarks are the column names of the landmark
        dataset without the _x/_y postfix.

        Args:
            
        Returns:
            all_lm_keys (list): List of landmark keys in the dataset. E.g. 
                'nose',... .
        """
        all_lm_keys = []
        for lm in self.lm_frame.columns:
            lm_key = re.sub('\_x', '', lm)
            if lm_key + '_x' == lm:
                all_lm_keys.append(lm_key)
        return all_lm_keys

    def _get_lm_keys(self):
        """Read and check landmark keys from options for output datasample.

        Returns:
            lm_keys (list): List of landmark keys for output datasample.
        """
        for lm_key in self.preprocess_lm_keys:
            if not lm_key in self.all_lm_keys:
                raise TypeError("""Landmark key {lm_key} in 
                        self.preprocess_lm_keys does not exist in CelebA 
                        dataset.""".format(lm_key=lm_key))
        
        if len(self.preprocess_lm_keys) == 0:
            return self.all_lm_keys

        return self.preprocess_lm_keys

    def _preprocess(self):
        """Builds up preprocessing pipeline from specified options.

        Returns:
            compose (obj): Torchvision.transforms.Compose object representing
                the preprocessing pipeline.
        """
        if len(self.preprocess) == 0:
            return None
        
        pipeline = []
        for step in self.preprocess:
            if not step in self.preprocess_dict.keys():
                raise ValueError("""Step {step} is not defined in 
                        preprocess_dict.""".format(step=step))
            processing = self.preprocess_dict[step]()
            pipeline.append(processing)

        return transforms.Compose(pipeline)

    def _preprocess_rescale(self):
        """Instantiate Rescale functor from options.
        
        Returns:
            rescale (obj): Instantiated Rescale functor object. 
        """
        output_size = self.rescale_output_size
        output_size = CelebaDataset.list2scalar_tuple(output_size)
        
        return Rescale(output_size, *self.lm_keys)

    def _preprocess_randcovercrop(self):
        """Instantiate RandCoverCrop functor from options.

        Returns:
            randcovercrop (obj): Instantiated RandCoverCrop functor object.
        """
        output_size = self.randcovercrop_output_size
        output_size = CelebaDataset.list2scalar_tuple(output_size)
        lms_covered = self.randcovercrop_lms_covered
        padding = self.randcovercrop_padding
        padding = CelebaDataset.list2scalar_tuple(padding)
        no_rand = self.randcovercrop_no_rand

        return RandCoverCrop(output_size, lms_covered, padding, no_rand, 
                *self.lm_keys)

    def _preprocess_randhorflip(self):
        """Instantiate RandHorFlip functor from options.

        Returns:
            horflip (obj): Instantiated RandHorFlip functor object.
        """
        p = self.randhorflip_p
        if not 0 <= p <= 1:
            raise ValueError('randhorflip_p must between 0 and 1.')

        return RandHorFlip(p, *self.lm_keys)

    def _preprocess_totensor(self):
        """Instantiate ToTensor functor.

        Returns:
            totensor (obj): Instantiated ToTensor object.
        """
        return ToTensor(*self.lm_keys)

    def _preprocess_normalize(self):
        """Instantiate Normalize functor from options.

        Returns:
            normalize (obj): Instantiated Normalize functor object.
        """
        mean = self.normalize_mean
        std = self.normalize_std

        return Normalize(mean, std, *self.lm_keys)

    
    @staticmethod
    def list2scalar_tuple(list_in):
        """Converts list of one element to a scalar and a list of two elements
        two a tuple.

        Args:
            list_in (list): List of one or two elements.

        Returns:
            scalar_tuple (scalar or tuple): Scalar of type inferred from type
                of list elements or tuple of two numbers, depending on length 
                of list.
        """
        scalar_tuple = 0
        
        if isinstance(list_in, (int, float)):
            scalar_tuple = list_in
            return scalar_tuple

        if len(list_in) == 1:
            scalar_tuple = list_in[0]
        elif len(scalar_tuple) == 2:
            scalar_tuple = list_in[0], list_in[1]
        else:
            raise TypeError("""List which should be converted to scalar or 
                    tuple must contain one or two elements but {n} elements are 
                    given.""".format(n=len(list_in)))
        
        return scalar_tuple

    

class BaseTrf(ABC):
    """Abstract base class for implementing data transformation functors for 
    CelebA dataset.

    Implements functionality for handling optional landmark keys.

    Attributes:
        lm_keys (list): List for specifying landmarks. If no landmarks are
            specified the landmarks are inferred from the data sample when the 
            functor is called for the first time.
    """
    def __init__(self, *lm_keys):
        """Initialization with option to select only certain landmarks.
        
        Args:
            *lm_keys (str): Optional number of keys in datasample for 
                selecting only specified landmarks. If no landmark keys are
                given all landmarks within data sample are considered.
                Landmark keys should given without coordinate postfix. E.g. 
                ['nose', 'lefteye'].
        """
        self.lm_keys = []
        for lm_key in lm_keys:
            self.lm_keys.append(lm_key)

    
    @abstractmethod
    def __call__(self):
        """Abstract method to make object of class callable (functor).
        """
        pass

    @staticmethod
    def update_lm_keys(func):
        """Decorator for __call__ method for updating attribute self.lm_keys 
        according to landmark keys given by sample.
        
        If self.lm_keys is empty all landmarks from sample are inferred.

        Args:
            sample (dict): Dictionary containing the data sample. Contains 
                the key 'img' and at least one landmark key as e.g. 'nose'.

        Returns:
        """
        @functools.wraps(func)
        def wrapper_update_lm_keys(self, sample):
            #if no landmark keys are specified use all landmarks within sample
            if len(self.lm_keys) == 0:
                for lm_key in sample.keys():
                    if not lm_key == 'img':
                        self.lm_keys.append(lm_key)
            
            #check validity of landmark keys
            if len(self.lm_keys) == 0:
                raise TypeError('Data sample does not contain landmarks.')
            
            for lm_key in self.lm_keys:
                if not lm_key in sample.keys():
                    raise TypeError("""Specified landmark {lm} is not within 
                            given data sample.""".format(lm=lm_key))
            
            #execute wrapped function
            value = func(self, sample)
            
            return value

        return wrapper_update_lm_keys
   

class ToTensor(BaseTrf):
    """Acts as functor and converts image and specified landmarks within sample 
    to tensor.

    Image dimensions are flipped from H x W x C to C x H x W and range is 
    squashed from [0,255] to [0.,1.0]. When no landmarks are specified, all 
    landmarks within the sample are considered. If no landmark keys are 
    specified in the constructor, all landmarks within the sample are 
    transformed. This functionality is handled by parent class BaseTrf. 
    Returned sample contains image data and specified landmark data.

    Attributes:
        lm_keys (str): Landmark keys within data sample for which numpy arrays 
            should be converted.     
    """

    def __init__(self, *lm_keys):
        """Initialization with option to select only certain landmarks. 

        Args:
            *lm_keys (str): Optional number of keys in datasample for 
                selecting only specified landmarks. If no landmark keys are
                given all landmarks within data sample are considered.
                Landmark keys should be given without coordinate postfix. E.g. 
                ['nose', 'lefteye'].
        """
        super(ToTensor, self).__init__(*lm_keys)
    
    @BaseTrf.update_lm_keys
    def __call__(self, sample):
        """Converts ndarrays within sample to tensors of type Float (not 
        DoubleFloat).

        Flips dimensions of image data from H x W x C to C x H x W and 
        transforms range [0,255] to [0.,1.0].

        Args:
            sample (dict): A sample containing numpy arrays.

        Returns:
            tensor_sample (dict): Converted sample containing tensors of type
                Float.
        """
        tensor_sample = {}    
        img = sample['img']
        img = img.transpose((2,0,1))
        tensor_sample['img'] = torch.from_numpy(img).float()
        for lm_key in self.lm_keys:
            lm = sample[lm_key]
            tensor_sample[lm_key] = torch.from_numpy(lm).float() 
        
        return tensor_sample

class RandHorFlip(BaseTrf):
    """This class implements an horizontal flip as a simple data augmentation
    technique. 

    The horizontal flip is performed with a probability of 0.5. If no 
    horizontal flip is performed the output sample is the unchanged input 
    sample.

    Attributes:
        *lm_keys (str): Landmark keys which should be in output data sample.
    """

    def __init__(self, p=0.5, *lm_keys):
        """Initialization with option to select only certain landmarks.

        Args:
            p (float): Probability for horizontal flip.
            *lm_keys (str): Optional number of keys in datasample for 
                selecting only specified landmarks. If no landmark keys are
                given all landmarks within data sample are considered.
                Landmark keys should be given without coordinate postfix. E.g. 
                ['nose', 'lefteye'].
        """
        super(RandHorFlip, self).__init__(*lm_keys)
        self.p = p

    @BaseTrf.update_lm_keys
    def __call__(self, sample):
        """Flips image horizontally.
        
        Args:
            sample (dict): A data sample with image and landmark data.

        Returns:
            flipped_sample (dict): Data sample with flipped image.
        """
        p_coin = (self.p, 1.- self.p)
        throw_coin = np.random.choice((0,1), p=p_coin)
        if throw_coin == 1:
            return sample

        flipped_sample = {}
        img = sample['img']
        flipped_img = img[:, ::-1,:]
        flipped_sample['img'] = flipped_img.copy()
        #landmarks
        width = img.shape[1]
        for lm_key in self.lm_keys:
            lm_x = width - 1 - sample[lm_key][0]
            lm_y = sample[lm_key][1]
            flipped_sample[lm_key] = np.array((int(lm_x), int(lm_y)))

        tmp_lefteye = flipped_sample['lefteye']
        flipped_sample['lefteye'] = flipped_sample['righteye']
        flipped_sample['righteye'] = tmp_lefteye

        tmp_leftmouth = flipped_sample['leftmouth']
        flipped_sample['leftmouth'] = flipped_sample['rightmouth']
        flipped_sample['rightmouth'] = tmp_leftmouth

        
        return flipped_sample



class Rescale(BaseTrf):
    """Acts as functor and rescales image and specified landmarks within 
    sample. Image is tranformed to range [0.,1.].

    When no landmarks are specified, all landmarks within the sample are
    considered. If no landmark keys are specified in the constructor, all 
    landmarks within the sample are transformed. This functionality is handled
    by parent class BaseTrf. Returned sample contains image data and specified 
    landmark data.

    Attributes:
        lm_keys (str): Landmark keys within data sample for which numpy arrays 
            should be converted.     
    
    """
    def __init__(self, output_size, *lm_keys):
        """Initialization with option to select only certain landmarks.

        Args:
            output_size (int or tuple): Output size of image. If output_size is
                tuple (h,w) it represents height and width of the output image.
                If output_size is int smaller part of height and width is
                rescaled and aspect ratio is kept the same.
            *lm_keys (str): Optional number of keys in datasample for 
                selecting only specified landmarks. If no landmark keys are
                given all landmarks within data sample are considered.
                Landmark keys should be given without coordinate postfix. E.g. 
                ['nose', 'lefteye'].
        """
        super(Rescale, self).__init__(*lm_keys)
        if not isinstance(output_size, (int, tuple)):
            raise TypeError('output_size for rescaling must be int or tuple.')
        self.output_size = output_size

    @BaseTrf.update_lm_keys
    def __call__(self, sample):
        """Rescales image and landmark data.

        Args:
            sample (dict): A data sample with image and landmark data.

        Returns:
            rescaled_sample (dict): Rescaled data sample.
        """
        
        rescaled_sample = {}
        img = sample['img']

        #img in format H x W x C
        h_in, w_in = img.shape[:2]
        if isinstance(self.output_size, int):
            if h_in > w_in:
                w_out = self.output_size
                h_out = h_in * w_out / w_in
            else:
                h_out = self.output_size
                w_out = w_in * h_out / h_in
        else:
            h_out, w_out = self.output_size
        
        #image
        img = transform.resize(img, (int(h_out), int(w_out)))
        rescaled_sample['img'] = img
        
        #landmarks
        for lm_key in self.lm_keys:
            lm_x = sample[lm_key][0] * w_out / w_in
            lm_y = sample[lm_key][1] * h_out / h_in
            rescaled_sample[lm_key] = np.array((int(lm_x), int(lm_y)))

        return rescaled_sample

class Crop(BaseTrf):
    """Acts as a functor and crops a subimage from specified rectangle region.

    Attributes:
        lm_keys (list): Landmark keys which should be contained in output
            sample. 
        rec (dict): Representation of rectangle region which should be cropped. 
            Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max'.
    """
    def __init__(self, rec, *lm_keys):
        """Initialization with option to define landmarks which should be added
        to cropped sample.

        Args:
            rec (dict): Representation of rectangle region which should be
                cropped. Dictionary with keys 'x_min', 'x_max', 'y_min', 
                'y_max'.
            *lm_keys (str): Optional number of keys in datasample for 
                selecting only specified landmarks in returned data sample. If 
                no landmark keys are given all landmarks within data sample are 
                considered. Landmark keys should be given without coordinate 
                postfix. E.g. ['nose', 'lefteye'].
        """

        super(Crop, self).__init__(*lm_keys)
        self.rec = rec

    @BaseTrf.update_lm_keys
    def __call__(self, sample):
        """Crops a specified rectangle region from image in sample and converts
        landmarks accordingly. 
        
        Args:
            sample (dict): A data sample with image and landmark data.

        Returns:
            cropped_sample (dict): Cropped data sample.
        """
        cropped_sample = {}

        #crop image
        img = sample['img']
        #img dimensions H x W x C
        cropped_img = img[
                self.rec['y_min']:self.rec['y_max'], 
                self.rec['x_min']:self.rec['x_max']
                ]
        cropped_sample['img'] = cropped_img

        #transform landmarks accordingly
        for lm_key in self.lm_keys:
            xy = sample[lm_key]
            x_cropped = xy[0] - self.rec['x_min']
            y_cropped = xy[1] - self.rec['y_min']
            xy_cropped = np.array((x_cropped, y_cropped))
            cropped_sample[lm_key] = xy_cropped

        return cropped_sample

class RandCoverCrop(BaseTrf):
    """Acts as a functor and crops a subimage with specified size by random 
    such that all specified landmarks are covered definitely. 
    
    For ensuring that all specified landmarks are covered a minimum rectangle 
    around the landmarks is created and extended by specified padding in x and 
    y dimension.

    Attributes:
        lms_covered (list): List with landmarks which should be covered by the
            random crop definitely.
        output_size_x (int): Width of output image.
        output_size_y (int): Heigth of output image.
        padding_x (float): Padding in x dimension given as fraction of input
            image width.
        padding_y (flaot): Padding in y dimension given as fraction of input
            image height.
        no_rand (boolean): If true, no random sampling for crop position.
            Especially for testing.
    """

    def __init__(self, output_size, lms_covered, padding=0., no_rand=False, 
            *lm_keys):
        """Initialization with option to define landmarks which should be 
        covered for sure and landmarks which should be in output sample.

        Args:
            output_size (int or tuple): Specifies width and height of output 
                image. If output_size is int width and height are equal in size
                and given by output_size. If outputsize is tuple first entry 
                defines width and second heigth of output image.
            lms_covered (list): Landmarks which are covered in cropped image 
                definitely. Landmarks should be specified by name without 
                coordinate postfix, e.g. 'nose'.
            padding_ratio (float or tuple): Specifies padding by fraction of 
                original image dimension. When padding is float both dimensions 
                uses same padding ratio. With a tuple padding rates can be set 
                for both dimension separately (padding_x, padding_y). When 
                padding whould lead to exceeding the image boundaries maximal 
                padding without this violation is applied.
            no_rand (boolean): If True, random sampling for crop position is 
                switched of and the most central crop position of the valid 
                range is used. In particular for reproducability in the testing
                phase.
            *lm_keys (str): Optional number of keys in datasample for 
                selecting only specified landmarks in returned data sample. If 
                no landmark keys are given all landmarks within data sample are 
                considered. Landmark keys should be given without coordinate 
                postfix. E.g. ['nose', 'lefteye'].
        """
        
        super(RandCoverCrop, self).__init__(*lm_keys)
        
        if isinstance(output_size, int):
            self.output_size_x = output_size
            self.output_size_y = output_size
        else:
            self.output_size_x = output_size[0]
            self.output_size_y = output_size[1]

        self.lms_covered = lms_covered
        
        if isinstance(padding, float):
            self.padding_x = padding
            self.padding_y = padding
        else:
            self.padding_x = padding[0]
            self.padding_y = padding[1]
        
        self.no_rand = no_rand

    @BaseTrf.update_lm_keys
    def __call__(self, sample):
        """Crops a subimage with specified size by random such that specified
        landmarks are covered definitely. 

        Args:
            sample (dict): A data sample with image and landmark data.

        Returns:
            cropped_sample (dict): Cropped data sample.
        """

        #rectangle region which should be covered definitely
        rec_covered = { 
                'x_min': 0,
                'x_max': 0,
                'y_min': 0,
                'y_max': 0 }

        for i, lm_cover in enumerate(self.lms_covered):
            if not lm_cover in sample.keys():
                raise ValueError("""Specified landmark {lm_cover} which should 
                        be covered by random crop does not exist in data 
                        sample.""".format(lm_cover=lm_cover))
        
        #update rectangle region by landmark positions
            x = sample[lm_cover][0]
            y = sample[lm_cover][1]
            if i==0:
                rec_covered['x_min'] = x
                rec_covered['x_max'] = x
                rec_covered['y_min'] = y
                rec_covered['y_max'] = y
            else:
                #update covered rectangle region
                if rec_covered['x_min'] > x:
                    rec_covered['x_min'] = x
                if rec_covered['x_max'] < x:
                    rec_covered['x_max'] = x
                if rec_covered['y_min'] > y:
                    rec_covered['y_min'] = y
                if rec_covered['y_max'] < y:
                    rec_covered['y_max'] = y
        
        #img dimensions H x W x C
        width = sample['img'].shape[1]
        height = sample['img'].shape[0]

        #extend rectangle region by padding 
        if not (0 <= self.padding_x <= 1 and 0 <= self.padding_y <= 1): 
            raise ValueError('Padding must be specified as ratio and be in the ' +
                    'interval [0., 1.].')
        pix_padding_x = int(width * self.padding_x)
        pix_padding_y = int(height * self.padding_y)
        #clip padding to image boundaries if necessary
        rec_covered['x_min'] = max(0, rec_covered['x_min'] - pix_padding_x)
        rec_covered['x_max'] = min(width - 1, rec_covered['x_max'] + pix_padding_x)
        rec_covered['y_min'] = max(0, rec_covered['y_min'] - pix_padding_y)
        rec_covered['y_max'] = min(height -1, rec_covered['y_max'] + pix_padding_y)
        
        crop_range_x = RandCoverCrop.crop_range_1d(
                width, 
                self.output_size_x, 
                rec_covered['x_min'],
                rec_covered['x_max']
                )
        
        crop_range_y = RandCoverCrop.crop_range_1d(
                height, 
                self.output_size_y, 
                rec_covered['y_min'],
                rec_covered['y_max']
                )
       
        ul_corner_x = 0
        ul_corner_y = 0

        if self.no_rand:
            ul_corner_x = int((crop_range_x[0] + crop_range_x[1]) / 2)
            ul_corner_y = int((crop_range_y[0] + crop_range_y[1]) / 2)
        else:
            #sample upper left corner of cropped subimage
            ul_corner_x = random.randint(
                    crop_range_x[0],
                    crop_range_x[1]
                    )

            ul_corner_y = random.randint(
                    crop_range_y[0],
                    crop_range_y[1]
                    )
        


        rec_crop = {
                'x_min': ul_corner_x,
                'x_max': ul_corner_x + self.output_size_x - 1,
                'y_min': ul_corner_y,
                'y_max': ul_corner_y + self.output_size_y - 1
                }

        cropping = Crop(rec_crop, *self.lm_keys)
        cropped_sample = cropping(sample)

        return cropped_sample
        

    @staticmethod
    def crop_range_1d(N, n, a, b):
        """Computes the allowed range for the starting point of a cropped image
        edge such that it is within original image edge and covers a specified 
        interval.
        
        Consider the original image edge of length N as an interval [0,N-1]. 
        Now the interval [a,b] which should be covered by the cropped image 
        edge is within [0,N-1]. The cropped image edge is of length n and has
        starting point s, i.e. it corresponds to the interval [s, s+n-1]. This
        method returns the maximum interval [l, u] such that cropped image 
        edges with a starting point s within this interval cover the given 
        interval [a,b] and are within original image edge interval [0, N-1].
        Sketch:

               [ s,... , a,...          , b, ..., s+n-1 ]
                       [ a,...          , b ] 
        [ 0,..., s,... , a,...          , b ,..., s+n-1,...         , N-1 ]

        Args:
            N (int): Length of the original image edge.
            n (int): Length of the cropped image edge.
            a (int): Lower bound of interval that should be covered.
            b (int): Upper bound of interval that should be covered.
        
        Returns:
            crop_range (obj): Range for starting point of cropped image edge. 
                crop_range is an ndarray of shape (2,) containing the lower and
                upper bound [l,u].
        """
        #if (a < 0 or b > N-1):
        #    raise ValueError('Specified interval [a,b] must be within [0,N].')
        #if b - a + 1 > n:
        #    raise ValueError("""Length of cropped image edge n must be greater 
        #            equal length of specified interval b-a+1.""")
        #if n > N:
        #    raise ValueError("""Length of cropped image edge n must be smaller 
        #            equal original image edge length N.""")
        
        #lower and upper bound from limitations of original image edge size
        l_N = 0
        u_N = N - 1 - (n - 1)
        #lower and upper bound for covering interval [a,b]
        l_cover = b - (n - 1)
        u_cover = a 

        l = max(l_N, l_cover)
        u = min(u_N, u_cover)
        #u = a
        crop_range = np.array((l,u))
        
        return crop_range

class Normalize(BaseTrf):
    """Acts as functor and normalizes the image.

    Normalizes the image value range according to 
    input[channel] = (input[channel] - mean[channel]) / std[channel]

    Attributes:
        lm_keys (list): Landmark keys which should be contained in output
            sample. 
        mean (seq): Sequence of mean values for each channel.
        std (seq): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
            *lm_keys):
        """Initialization with option to select landmark keys in output sample.

        Specify the mean and standard deviation and landmarks for output_sample.

        Args:
            mean (seq): Sequence of mean values for each channel. Default
                values for using pretrained torch models.
            std (seq): Sequence of standard deviations for each channel.
                Default values for using pretrained torch models.
            *lm_keys (str): Optional number of keys in datasample for 
                selecting only specified landmarks in returned data sample. If 
                no landmark keys are given all landmarks within data sample are 
                considered. Landmark keys should be given without coordinate 
                postfix. E.g. ['nose', 'lefteye'].
        """
        super(Normalize, self).__init__(*lm_keys)
        self.mean = mean
        self.std = std
    
    @BaseTrf.update_lm_keys
    def __call__(self, sample):
        """Normalizes the image according to specified mean and std.
    
        Normalizes the image value range according to 
        input[channel] = (input[channel] - mean[channel]) / std[channel]
        
        Args:
            sample (dict): Data sample with image and landmark data.

        Returns:
            normalized_sample (dict): Data sample with noramlized image.
        """

        normalized_sample = {}

        #image
        img = sample['img']
        img = transforms.Normalize(self.mean, self.std)(img)
        normalized_sample['img'] = img

        #landmarks
        for lm_key in self.lm_keys:
            normalized_sample[lm_key] = sample[lm_key]

        return normalized_sample



