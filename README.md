# MaterialGAN: Reflectance Capture using a Generative SVBRDF Model

[Yu Guo](https://tflsguoyu.github.io/), Cameron Smith, [Miloš Hašan](http://miloshasan.net/), [Kalyan Sunkavalli](http://www.kalyans.org/) and [Shuang Zhao](https://shuangz.com/).

In ACM Transactions on Graphics (SIGGRAPH Asia 2020).

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/github/teaser.jpg" width="1000px">

[[Paper](https://github.com/tflsguoyu/materialgan_paper/blob/master/materialgan.pdf)]
[[Code](https://github.com/tflsguoyu/svbrdf-diff-renderer)]
[[Supplemental Materials](https://tflsguoyu.github.io/materialgan_suppl/)]
[[Poster](https://github.com/tflsguoyu/materialgan_poster/blob/master/materialgan_poster.pdf)]
[Fastforward on Siggraph Asia 2020 ([Video](https://youtu.be/fD6CTb1DlbE))([Slides](https://www.dropbox.com/s/qi594y27dqa7irf/materialgan_ff.pptx?dl=0))] \
[Presentation on Siggraph Asia 2020 ([Video](https://youtu.be/CrAoVsJf0Zw))([Slides](https://www.dropbox.com/s/zj2mhrminoamrdg/materialgan_main.pptx?dl=0))]

## Python dependencies [torch, torchvision, opencv-python, matplotlib, pupil_apriltags]
Tested on 
1. MacOS, python3.11, pytorch2.2(CPU)
2. Windows10, python3.11, pytorch2.3, CUDA11.8 

## Pretrained MaterialGAN model
Download all the checkpoints to the folder `ckp`: 
[`materialgan.pth`](https://www.dropbox.com/scl/fi/z41e6tedyh7m57vatse7p/materialgan.pth?rlkey=ykovb3owafmz6icvss13sdddl&dl=0)
[`latent_avg_W+_256.pt`](https://www.dropbox.com/scl/fi/nf4kfoiqx6h7baxpbfu01/latent_avg_W-_256.ptrlkey=ot0yfkbgq47vt45huh65mgwit&st=724ubgqp&dl=0)
[`latent_const_W+_256.pt`](https://www.dropbox.com/scl/fi/mdh8boshpfc6lwktrfh4i/latent_const_W-_256.pt?rlkey=gy55tp5h6c91icxhdzzbf5sss&st=hzxk2580&dl=0)
[`latent_const_N_256.pt`](https://www.dropbox.com/scl/fi/320aov4ahc4wkhaq8mpve/latent_const_N_256.pt?rlkey=ckydqxdpyvzy7kns2h0geuh4e&st=d7ytmxz5&dl=0)

## Usage
To optimize SVBRDF maps, we need several images with different lighting and a corresponding JSON file, which has all the information included. 
If you use our dataset, all the JSON files are provided. If you want to capture new data, see below instruction. The JSON file will be generated automatically.

See `run.py` for more details. 

## Capture your own data with a smartphone
1. Print "fig/tag36h11_print.png" on a solid paper with proper size and crop the center area.
2. Measure `size`(in cm unit) with a ruler, see the red arrow line in the below figure.
3. Place it on the material you want to capture and make the paper as flat as possible.
4. Turn on the camera flashlight and capture images from different views.
5. Create a data folder for captured images. We provide an example here, `data/yellow_box-17.0-0.1/raw`.
6. Run `gen_targets_from_capture(Path("data/yellow_box-17.0-0.1"), size=17.0, depth=0.1)` in `run.py`.
The `size` here is the number you measured from step 2; `depth` is the distance (in cm unit) between the marker plane and the material plane. For example, if you attach the markers on thick cardboard, you should use a larger `depth`.
7. The generated target images are located in `data/yellow_box-17.0-0.1/target` and the corresponding JSON files are generated as well.
<img src="https://github.com/tflsguoyu/svbrdf-diff-renderer/blob/master/fig/fig1.png" width="600px">

Tips:
1. All markers should be captured and in focus and the letter `A` should be facing up.
2. It's better to capture during the night or in a dark room and turn off other lights.
3. It's better to see the highlights in the cropped area.
4. Change camera mode to manual, and keep the white balance and focal length the same during the captures.
5. `.heic` image format is not supported now. Convert it to `.png`/`.jpg` first. 
6. Preferred capturing order: highlight in topleft -> top -> topright -> left -> center -> right -> bottomleft -> bottom -> bottomright. See images in `data/yellow_box/raw` as references.

## The real data we used in the paper [[Download](https://drive.google.com/file/d/1Vs2e35c4bNHRUu3ON4IsuOOP6uK8Ivji/view?usp=sharing)]
The dataset includes corresponding JSON files. We put our results here as a reference, and you can also generate the results using our code from `run.py`.
- `optim_ganlatent(material_dir / "optim_latent_256.json", 256, 0.02, [2000, 10, 10], ["ckp/latent_avg_W+_256.pt"])`
- `optim_perpixel(material_dir / "optim_pixel_256_to_512.json", 512, 0.01, 100, tex_init="textures")`
- `optim_perpixel(material_dir / "optim_pixel_512_to_1024.json", 1024, 0.01, 100, tex_init="textures")`

For most of the cases, we use `ckp = ["ckp/latent_avg_W+_256.pt"]` as the initialization, as shown below.

From left to right: input photos, output texture maps (256x256) from MaterialGAN, output high-res maps (1024x1024) from per-pixel optimization.

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bathroomtile1/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bathroomtile1/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bathroomtile1/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bathroomtile2/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bathroomtile2/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bathroomtile2/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/book1/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/book1/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/book1/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/book2/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/book2/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/book2/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/cards-blue/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/cards-blue/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/cards-blue/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/cards-red/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/cards-red/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/cards-red/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag1/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag1/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag1/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag2/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag2/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag2/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag3/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag3/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/giftbag3/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-blue/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-blue/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-blue/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-brown/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-brown/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-brown/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-darkbrown/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-darkbrown/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/leather-darkbrown/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-carpet/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-carpet/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-carpet/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-foam/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-foam/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-foam/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-red-carton/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-red-carton/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-red-carton/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-red-grid/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-red-grid/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-red-grid/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/rubber-pattern/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/rubber-pattern/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/rubber-pattern/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-bathroom-tile/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-bathroom-tile/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-bathroom-tile/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-bigtile/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-bigtile/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-bigtile/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-smalltile/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-smalltile/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-smalltile/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-granite/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-granite/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-granite/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-ground-flake/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-ground-flake/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-ground-flake/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-shiny/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-shiny/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-shiny/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-vinyl-floor/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-vinyl-floor/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-vinyl-floor/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-color/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-color/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-color/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-red-bump/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-red-bump/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-red-bump/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-plaster-green/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-plaster-green/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-plaster-green/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-plaster-white/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-plaster-white/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wall-plaster-white/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-alder/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-alder/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-alder/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-jatoba/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-jatoba/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-jatoba/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-knotty/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-knotty/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-knotty/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-laminate/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-laminate/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-laminate/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-t/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-t/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-t/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-tile/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-tile/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-tile/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-treeskin/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-treeskin/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-treeskin/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-walnut/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-walnut/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-walnut/optim_latent/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-bamboo/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-bamboo/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-bamboo/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bamboo-veawe/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bamboo-veawe/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/bamboo-veawe/optim_latent/1024/tex.jpg" width="150px">

<!-- <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/VVVVVVVVVVV/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/VVVVVVVVVVV/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/VVVVVVVVVVV/optim_latent/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/VVVVVVVVVVV/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/VVVVVVVVVVV/optim_latent/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/VVVVVVVVVVV/optim_latent/1024/tex.jpg" width="150px"> -->

For some specular materials, you can see the highlights are bakes in the roughness maps. You could try different initialization like `ckp = ["ckp/latent_const_W+_256.pt", "ckp/latent_const_N_256.pt"]`, which use lower roughness as initial. See the results below, which converged to better results.

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-granite/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-granite/optim_latent_const/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-granite/optim_latent_const/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-ground-flake/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-ground-flake/optim_latent_const/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-ground-flake/optim_latent_const/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-shiny/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-shiny/optim_latent_const/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-shiny/optim_latent_const/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-vinyl-floor/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-vinyl-floor/optim_latent_const/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/stone-spec-vinyl-floor/optim_latent_const/1024/tex.jpg" width="150px">

<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-red-carton/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-red-carton/optim_latent_const/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/plastic-red-carton/optim_latent_const/1024/tex.jpg" width="150px">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-bamboo/target/all.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-bamboo/optim_latent_const/256/tex.jpg" width="150px"> <img src="https://github.com/tflsguoyu/materialgan_suppl/blob/master/data/wood-bamboo/optim_latent_const/1024/tex.jpg" width="150px">

Notes, the high-res output uses MaterialGAN output as the initial but only has pixel loss constrain during the optimization. Although more details are recovered, sometimes it will overfit. See the above example.

We will provide more data in the future.
