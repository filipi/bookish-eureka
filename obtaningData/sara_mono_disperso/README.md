
Convert the tranparent background to white

```ls -1 *.png |  cut -d . -f 1 | while read line; do convert -flatten $line.png $line.png ; done;

Convert images to a movie

```ffmpeg -i simpson_1972_%02d.png simpson_2917.mpg

Cropp to remove scale

```ls -1 *.png | cut -d . -f 1 | while read line ; do convert .png -crop 891x217+52+23 cropped/.mini.png; done

White patch iso-contours which are not the front

```convert phixz13_0_usf.mini.png -strokewidth 0 -fill "rgba( 255, 255, 255 , 1 )" -draw "rectangle 2,100 50,230 " ./phixz13_0_usf.mini.png 



Cropp to remove scale

```for f in *.png; do echo convert "$f" -crop  890x217+52+23  "${f/.png/-cropped.png}"; done;```

Red patch iso-contours which are not the front to test

```convert phixz70_usf-cropped.png -strokewidth 0 -fill "rgba( 255, 0, 0 , 1 )" -draw "rectangle 60,60 167,0 " output.png```

White patch iso-contours which are not the front to test

for f in *.png; do convert "$f" -strokewidth 0 -fill "rgba( 255, 255, 255 , 1 )" -draw "rectangle 450,217 0,0 " "${f/.png/-cleaned.png}"; done;

Renumbering files:

```ls | cat -n | while read n f; do mv "$f" `printf "phixz_usg_%02d.png" $n`; done```

Generate video

``` ffmpeg -framerate 12 -i phixz_usg_%2d.png -c:v libx264 -r 120 output.mp4```

