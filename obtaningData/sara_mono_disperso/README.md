
Convert the tranparent background to white

```ls -1 *.png |  cut -d . -f 1 | while read line; do convert -flatten $line.png $line.png ; done;

Convert images to a movie

```ffmpeg -i simpson_1972_%02d.png simpson_2917.mpg

Cropp to remove scale

```ls -1 *.png | cut -d . -f 1 | while read line ; do convert .png -crop 891x217+52+23 cropped/.mini.png; done

Withe patch iso-contours which are not the front

```convert phixz13_0_usf.mini.png -strokewidth 0 -fill "rgba( 255, 255, 255 , 1 )" -draw "rectangle 2,100 50,230 " ./phixz13_0_usf.mini.png 