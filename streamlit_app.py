import streamlit as st
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import models
import yaml
from skimage import io, transform, filters
import cv2 as cv
from scipy import ndimage
from pathlib import Path
from argparse import Namespace
from PIL import Image, ImageFilter

from options.opt_manager import OptManager
from utility.plotting import polygon_on_img


st.beta_set_page_config(
        page_title="ConstraintNet",
        #page_icon="",
        layout="centered",
        initial_sidebar_state="collapsed"
        )


# read state of last run
widget_values = {}
dest = Path('./utility/widget_values.yaml')
with dest.open('r') as f:
    widget_values = yaml.load(f)



st.write("""
        # Facial Landmark Detection with ConstraintNet
        """)

################################
#LOAD CONFIG####################
################################

#st.write("""
#        ## Load Config
#        """)


def file_selector_config(widget_values, folder_path=Path('./experiments/')):
    filenames = [f for f in folder_path.glob('**/config_test*.yaml') if not
        'sector_of_a_circle' in str(f)]
    index = 0
    for i, filename in enumerate(filenames):
        if Path(widget_values['file_config']) == filename:
            index = i
            break
    selected_filename = st.selectbox('Select config file:', filenames, index=index)
    return selected_filename


@st.cache
def dict2opts(opts_dict):
    opts = {}
    for blocks, block_opts in opts_dict['options'].items():
        for opt, value in block_opts.items():
            opts[opt] = value
    opts_ns = Namespace(**opts)
    return opts_ns


opts = 0
with st.beta_expander('0 Load Config', expanded=True):
    file_config = Path(widget_values['file_config'])
    file_config = file_selector_config(widget_values)
    widget_values['file_config'] = str(file_config)

    opts_dict = OptManager.yaml2dict(file_config)
    opts = dict2opts(opts_dict)

    st.write("Config Overview ")
    st.write("* Comment: ", opts.comment)
    st.write("* Model module: ", opts.model_module)
    if hasattr(opts, 'opts2constr_guard_layer'):
        st.write("* Constraint guard layer: ", opts.opts2constr_guard_layer)
    st.write("* Weight file: ", opts.reload_ckpt_file)
    
    show_config = widget_values['show_config']
    show_config_b = st.button('Show/Hide Total Config')
    if show_config_b:
        show_config = not show_config
    if show_config:
        st.json(opts_dict['options'])
    widget_values['show_config'] = show_config
   



################################
#LOAD IMAGE AND RESIZE##########
################################

#st.write("""
#        ## Load Image and Resize
#        """)


def file_selector_pic(folder_path=Path('./pics/')):
    filenames = [f for f in folder_path.iterdir() if not '.txt' in str(f)]
    index = 0
    for i, filename in enumerate(filenames):
        if Path(widget_values['file_pic']) == filename:
            index = i
            break
    selected_filename = st.selectbox('Select image file:', filenames) #,index=index)
    widget_values['file_pic'] = str(selected_filename)
    return selected_filename

def resize_img(img):
    #img in format H x W x C
    h_in, w_in = img.shape[:2]
    output_size = 300
    if h_in > w_in:
        w_out = output_size
        h_out = h_in * w_out / w_in
    else:
        h_out = output_size
        w_out = w_in * h_out / h_in

    #convert to PIL
    img = Image.fromarray(img)
    #from torchvision
    resize_it = transforms.Resize((int(h_out), int(w_out)))
    img = resize_it(img)
    img = np.array(img)

    return img

img = 0
#Path for image


img_file = Path(widget_values['file_pic'])
img = io.imread(img_file)
# resize
img = resize_img(img)
img_h = img.shape[0]
img_w = img.shape[1]

with st.beta_expander('1 Load Image and Resize', expanded=True):
    col_1_load, col_2_load = st.beta_columns(2)
    with col_1_load:
        img_file = file_selector_pic()
        img = io.imread(img_file)
        img_h = img.shape[0]
        img_w = img.shape[1]

        st.write(""" (height, width) = 
                """, (img_h, img_w))


        # resize
        img = resize_img(img)
        img_h = img.shape[0]
        img_w = img.shape[1]
        # info
        st.write("After resizing:")
        st.write("""(height, width) = 
                """, (img_h, img_w))



    with col_2_load:
        st.image(polygon_on_img(img), caption="Loaded image.", width=300)

    
################################
#CROP IMAGE#####################
################################

#st.write("""
#        ## Crop Image
#        """)

sl_x_min = widget_values['sl_x_min']
sl_y_min = widget_values['sl_y_min']

with st.beta_expander('2 Crop Image', expanded=True):
    st.write("""
            For cropping a 224 x 224 patch of the image,
            set top left corner of the rectangle:
            """)

    col_1_crop, col_2_crop = st.beta_columns(2)

    with col_1_crop:
        center = st.button('Center Crop')
        if center:
            sl_x_min = int((img_w - 224) / 2)
            sl_y_min = int((img_h - 224) / 2)
        sl_x_min = st.slider("x_min", 0, img_w-224, sl_x_min)
        widget_values['sl_x_min'] = sl_x_min
        sl_x_max = sl_x_min + 224 
        sl_y_min = st.slider("y_min", 0, img_h-224, sl_y_min)
        widget_values['sl_y_min'] = sl_y_min
        sl_y_max = sl_y_min + 224 

    top_left = np.array([sl_x_min, sl_y_min])
    top_right = np.array([sl_x_max, sl_y_min])
    bottom_right = np.array([sl_x_max, sl_y_max])
    bottom_left = np.array([sl_x_min, sl_y_max])

    polygon_crop = np.array([top_left, top_right, bottom_right, bottom_left])

    img_covered_area = polygon_on_img(img, polygon_crop, linewidth=5)

    with col_2_crop:
        st.image(img_covered_area, caption="This rectangle patch is cropped.", width=300)


rec = {'x_min': sl_x_min, 'y_min': sl_y_min}
img = img[
        rec['y_min']:rec['y_min']+224, 
        rec['x_min']:rec['x_min']+224
        ]


################################
#ADD A RECTANGLE################
################################

#st.write("""
#        ## Add a Rectangle
#        """)


def make_recording_widget(f):
    """Return a function that wraps a streamlit widget and records the
    widget's values to a global dictionary.
    """
    def wrapper(label, *args, **kwargs):
        widget_value = f(label, *args, **kwargs)
        widget_values[label] = widget_value
        return widget_value

    return wrapper



add_rec = widget_values['add_rec']
b_rec_x_min = widget_values['b_rec_x_min']
b_rec_x_max = widget_values['b_rec_x_max']
b_rec_y_min = widget_values['b_rec_y_min']
b_rec_y_max = widget_values['b_rec_y_max']
color_r = widget_values['color_r']
color_g = widget_values['color_g']
color_b = widget_values['color_b']


with st.beta_expander('3 Add a Rectangle'):
    add_rem = st.button('Add/Remove')
    if add_rem:
        add_rec = not add_rec
    if add_rec:
        col_1_rec, col_2_rec = st.beta_columns(2)
        with col_1_rec:
            st.write("Set left boundary")
            b_rec_x_min = st.slider("x_min", 0, 223, b_rec_x_min)
            widget_values['b_rec_x_min'] = b_rec_x_min
            st.write("Set top boundary")
            b_rec_y_min = st.slider("y_min", 0, 223, b_rec_y_min)
            widget_values['b_rec_y_min'] = b_rec_y_min
            
            st.write("Set Color")
            color_r = st.slider("R", 0, 255, color_r)
            widget_values['color_r'] = color_r
            color_g = st.slider("G", 0, 255, color_g)
            widget_values['color_g'] = color_g
            color_b = st.slider("B", 0, 255, color_b)
            widget_values['color_b'] = color_b
            
        with col_2_rec:
            st.write("Set right boundary")
            b_rec_x_max = st.slider("x_max", 0, 223, b_rec_x_max)
            widget_values['b_rec_x_max'] = b_rec_x_max

            st.write("Set bottom boundary")
            b_rec_y_max = st.slider("y_max", 0, 223, b_rec_y_max)
            widget_values['b_rec_y_max'] = b_rec_y_max
            img[b_rec_y_min:b_rec_y_max, b_rec_x_min:b_rec_x_max,0] = color_r
            img[b_rec_y_min:b_rec_y_max, b_rec_x_min:b_rec_x_max,1] = color_g
            img[b_rec_y_min:b_rec_y_max, b_rec_x_min:b_rec_x_max,2] = color_b
        
            img_rec = polygon_on_img(img)
            st.image(img_rec, caption="Added rectangle.", width=450)


widget_values['add_rec'] = add_rec

################################
#ROTATE IMAGE###################
################################

#st.write("""
#        ## Rotate Image 
#        """)

with st.beta_expander('4 Rotate Image'):
    col_1_rot, col_2_rot = st.beta_columns(2)
    rot_angle = widget_values['rot_angle']
    with col_1_rot:
        st.write("Set angle for rotation:")
        alpha_0 = st.button('No Rotation')
        if alpha_0:
            rot_angle = 0
        rot_angle = st.slider("alpha", -180, 180, rot_angle)
        widget_values['rot_angle'] = rot_angle
    with col_2_rot:
        img = Image.fromarray(img)
        img = transforms.functional.rotate(img, rot_angle)
        img = np.array(img)
        img_rotate = polygon_on_img(img)
        st.image(img_rotate, caption="Rotated image.", width=300)

################################
#BLURR IMAGE####################
################################

#st.write("""
#        ## Add Gaussian Blurring
#        """)

with st.beta_expander('5 Add Gaussian Blurring'):
    col_1_gauss, col_2_gauss = st.beta_columns(2)
    sigma = widget_values['sigma']
    with col_1_gauss:
        st.write("Set standard deviation:")
        sigma_0 = st.button('No Blurring')
        if sigma_0:
            sigma = 0
        sigma = st.slider('standard deviation', 0, 10, sigma, step=1)
        img = Image.fromarray(img)
        img = img.filter(ImageFilter.GaussianBlur(sigma))
        img = np.array(img)
        widget_values['sigma'] = sigma
    with col_2_gauss:
        img_blur = polygon_on_img(img)
        st.image(img_blur, caption="Blurred image.", width=300)


################################
#Set Constraints################
################################

img_for_plot = img
#ready for NN
img = img.transpose((2,0,1))
img = torch.from_numpy(img).float()
mean = opts.normalize_mean
std = opts.normalize_std
img = transforms.Normalize(mean, std)(img)




data = {}
data['img'] = img.unsqueeze(0)

if "triangle" in str(file_config):
    #st.write("""
    #            ## Set Triangle Constraint
    #            """)
    with st.beta_expander('6 Set Triangle Constraint'):
        sl_x_1 = widget_values['sl_x_1']
        sl_y_1 = widget_values['sl_y_1']
        sl_x_2 = widget_values['sl_x_2']
        sl_y_2 = widget_values['sl_y_2']
        sl_x_3 = widget_values['sl_x_3']
        sl_y_3 = widget_values['sl_y_3']
        col_1_constr_top, col_2_constr_top = st.beta_columns(2)
        with col_1_constr_top:
            st.write("Top vertex:")
            sl_x_1 = st.slider("x_t", 0, 223, sl_x_1)
            widget_values['sl_x_1'] = sl_x_1
            sl_y_1 = st.slider("y_t", 0, 223, sl_y_1)
            widget_values['sl_y_1'] = sl_y_1
        col_1_constr, col_2_constr = st.beta_columns(2)
        with col_1_constr:
            st.write("Bottom left vertex:")
            sl_x_3 = st.slider("x_bl", 0, 223, sl_x_3)
            widget_values['sl_x_3'] = sl_x_3
            sl_y_3 = st.slider("y_bl", 0, 223, sl_y_3)
            widget_values['sl_y_3'] = sl_y_3
        with col_2_constr:
            st.write("Bottom right vertex:")
            sl_x_2 = st.slider("x_br", 0, 223, sl_x_2)
            widget_values['sl_x_2'] = sl_x_2
            sl_y_2 = st.slider("y_br", 0, 223, sl_y_2)
            widget_values['sl_y_2'] = sl_y_2
            
        data['constr_para'] = torch.tensor([sl_x_1, sl_y_1, sl_x_2, sl_y_2, sl_x_3,
            sl_y_3]).float().unsqueeze(0)

        polygon_triangle = np.array([[sl_x_1, sl_y_1], [sl_x_2, sl_y_2], [sl_x_3, sl_y_3]])
        with col_2_constr_top:
            st.image(polygon_on_img(img_for_plot, polygon_triangle), width=300)


if ("exp_1" in str(file_config) and "bb" in str(file_config)):
    #st.write("## Set Bounding Box Constraints")
    with st.beta_expander('6 Set Bounding Box Constraints', expanded=True):
        col_1_bb, col_2_bb = st.beta_columns(2)
        sl_x_min_bb = widget_values['sl_x_min_bb']
        sl_y_min_bb = widget_values['sl_y_min_bb']
        sl_x_max_bb = widget_values['sl_x_max_bb']
        sl_y_max_bb = widget_values['sl_y_max_bb']
        with col_1_bb:
            st.write("Top left:")
            sl_x_min_bb = st.slider("x_min_bb", 0, 223, sl_x_min_bb)
            widget_values['sl_x_min_bb'] = sl_x_min_bb 
            sl_y_min_bb = st.slider("y_min_bb", 0, 223, sl_y_min_bb)
            widget_values['sl_y_min_bb'] = sl_y_min_bb 
        with col_2_bb:
            st.write("Bottom right:")
            sl_x_max_bb = st.slider("x_max_bb", 0, 223, sl_x_max_bb)
            widget_values['sl_x_max_bb'] = sl_x_max_bb 
            sl_y_max_bb = st.slider("y_max_bb", 0, 223, sl_y_max_bb)
            widget_values['sl_y_max_bb'] = sl_y_max_bb 
        
        data['constr_para'] = torch.tensor([sl_x_min_bb, sl_x_max_bb, sl_y_min_bb,
            sl_y_max_bb]).float().unsqueeze(0)

        polygon_bb = np.array([[sl_x_min_bb, sl_y_min_bb], [sl_x_max_bb,
            sl_y_min_bb], [sl_x_max_bb, sl_y_max_bb], [sl_x_min_bb, sl_y_max_bb]])


################################
#Apply ConstraintNet############
################################
#st.write("""
#        ## Detection
#        """)
pred_name = '6 Detection'
if ("exp_1" in str(file_config) and "bb" in str(file_config)) or ("triangle" in
        str(file_config)):
    pred_name = '7 Detection'

with st.beta_expander(pred_name, expanded=True):
    #load model from opts
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @st.cache
    def create_model():
        model = models.my_model(opts)
        model.to(device) 
        ckpt = torch.load(Path(opts.reload_ckpt_file))
        model.load_state_dict(ckpt)
        return model.eval()

    model = create_model()

    x_img = data['img'].to(device)

    if opts.model_module == 'constraintnet':
        constr_para = data['constr_para'].to(device)
        y_pred = model(x_img, constr_para)
    else:
        y_pred = model(x_img)


    if "exp_1" in str(file_config):     
        idx_x_nose = opts.lm_ordering_lm_order.index('nose_x')
        x_nose = y_pred.cpu().data[0, idx_x_nose]
        idx_y_nose = opts.lm_ordering_lm_order.index('nose_y')
        y_nose = y_pred.cpu().data[0, idx_y_nose]
        idx_x_lefteye = opts.lm_ordering_lm_order.index('lefteye_x')
        x_lefteye = y_pred.cpu().data[0, idx_x_lefteye]
        idx_y_lefteye = opts.lm_ordering_lm_order.index('lefteye_y')
        y_lefteye = y_pred.cpu().data[0, idx_y_lefteye]
        idx_x_righteye = opts.lm_ordering_lm_order.index('righteye_x')
        x_righteye = y_pred.cpu().data[0, idx_x_righteye]
        idx_y_righteye = opts.lm_ordering_lm_order.index('righteye_y')
        y_righteye = y_pred.cpu().data[0, idx_y_righteye]

        scatter_pts = (np.array([[x_nose, y_nose], [x_lefteye, y_lefteye],
            [x_righteye, y_righteye]]), 300 ,'x','white')

        if "resnet" in str(file_config):
            img_exp_1_resnet = polygon_on_img(img_for_plot, scatter_pts=scatter_pts)
            #io.imsave(Path('./pics/exp_1_resnet.jpg'),
            #        np.array(img_exp_1_resnet))
            st.image(img_exp_1_resnet, caption="Prediction with resnet without constraints", width=600)
        elif "bb_rel" in str(file_config):
            img_exp_1_bb_rel = polygon_on_img(img_for_plot, polygon_xy = polygon_bb,
                    linewidth=5, scatter_pts = scatter_pts)
            #io.imsave(Path('./pics/exp_1_bb_rel.jpg'),
            #        np.array(img_exp_1_resnet))
            st.image(img_exp_1_bb_rel, 
                    caption="Prediction within bb constraint and additional relative constraints", 
                    width=600)
        else:
            img_exp_1_bb = polygon_on_img(img_for_plot, polygon_xy = polygon_bb,
                    linewidth=5, scatter_pts = scatter_pts)
            #io.imsave(Path('./pics/exp_1_bb.jpg'),
            #        np.array(img_exp_1_bb))
            st.image(img_exp_1_bb, caption="Prediction within bb constraint.", width=600)



    if "exp_2" in str(file_config):
        x_nose = y_pred.cpu().data[0,0]
        y_nose = y_pred.cpu().data[0,1]

        scatter_pts = (np.array([[x_nose, y_nose]]), 300 ,'x','white')

        
        if "triangle" in str(file_config):
            img_exp_2_triangle = polygon_on_img(img_for_plot, polygon_xy = polygon_triangle,
                    linewidth=5, scatter_pts = scatter_pts)
            st.image(img_exp_2_triangle, caption="Prediction within triangle constraint", width=600)
            io.imsave(Path('./pics/exp_2_triangle.jpg'),
                    np.array(img_exp_2_triangle))
        else:
            img_exp_2_resnet = polygon_on_img(img_for_plot,
                    linewidth=5, scatter_pts = scatter_pts)
            #io.imsave(Path('./pics/exp_2_resnet.jpg'),
            #        np.array(img_exp_2_resnet))
            st.image(img_exp_2_resnet, 
                    caption="Prediction within ResNet50 (without constraints)", width=600)




with dest.open('w') as f:
    yaml.dump(widget_values, f)



