# ObjMST

conda env create -f stylegan3/environment.yml

conda activate ObjMST

pip install -r requirements.txt

1. Generating Consistent Style Representations
# model settings
stylegan3_path="/path/to//stylegan3" # StyleGAN3 repo path
stylegan3_pkl="/path/to//stylegan3/models/wikiart-1024-stylegan3-t-17.2Mimg.pkl" # StyleGAN3 wikiart pre-trained model path
stya2k_path="/path/to//StyA2K" # AdaAttN repo path

# experiment settings
exp_name="mask_ObjMST" # experiment name, which determines the output folder name

# set up the content images
content_paths="/path/to//Input/content/" # or use a folder / multiple folders of content images


# set up the multimodal style inputs
sty_text="Copper plate engraving" # the style text description
sty_img="/path/to/input/styles/contrast_of_forms.jpg" # the source style image

# Fisrt, generate the style representations
# alpha_text are the weights for the style text
# alpha_img are the weights for the style image


MKL_THREADING_LAYER=GNU python source_code/fg_stylization/gen_style_reps_fgbg.py \
    --exp_name $exp_name \
    --sty_text "$sty_text" \
    --sty_img $sty_img \
    --alpha_text 500 \
    --alpha_img 500 \
    --stylegan3_path $stylegan3_path \
    --stylegan3_pkl $stylegan3_pkl 
2. Foreground Stylization

3. Background Stylization

4. 
