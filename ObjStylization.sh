# model settings
stylegan3_path="/path/to/source_code/fg_stylization/stylegan3" # StyleGAN3 repo path
stylegan3_pkl="/path/to/source_code/fg_stylization/stylegan3/models/wikiart-1024-stylegan3-t-17.2Mimg.pkl" # StyleGAN3 wikiart pre-trained model path
stya2k_path="/path/to/source_code/fg_stylization/StyA2K" # AdaAttN repo path

# experiment settings
exp_name="mask_ObjMST" # experiment name, which determines the output folder name

# set up the content images
# content_paths="./contents/avril.jpg ./contents/cornell.jpg ./contents/modern.jpg" # use multiple content images
content_paths="/path/to/Input/Images/" # or use a folder / multiple folders of content images


# set up the multimodal style inputs
sty_text="Copper plate engraving" # the style text description
sty_img="/home/phdcs2/Hard_Disk/Projects/T2I/mmist/input/styles/contrast_of_forms.jpg" # the source style image

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


# Then, apply the style representations to the content images to generate Foreground Stylized Output
python source_code/fg_stylization/apply_style_reps_fgmaskstya2k.py \
    --style_reps_dir outputs_fgbg/$exp_name/style_reps \
    --output_dir outputs_fgbg/$exp_name/fg_stylized_imgs \
    --content_paths $content_paths \
    --stya2k_path $stya2k_path

# For single Text condition Output, 

1) #Train Input image using SinGAN

python source_code/bg_stylization/main_train.py --input_name "sailboat.png"

2) # Harmonize above input image as background on foreground Stylized Output
python source_code/bg_stylization/harmonization.py --input_name 'sailboat.png' --ref_name "fg_stylized_imgs/Copperplateengraving+contrast-of-forms_sailboat.png" --harmonization_start_scale 8


# For Double Text Condition Stylized Output :

1) # Train Background image using SinGAN
python source_code/bg_stylization/main_train.py --input_name "Desert_Sand.png"

2) # Harmonize above trained background image on foreground Stylized Output
python source_code/bg_stylization/harmonization.py --input_name 'sailboat.png' --ref_name "fg_stylized_imgs/Copperplateengraving+contrast-of-forms_sailboat.png" --harmonization_start_scale 8

