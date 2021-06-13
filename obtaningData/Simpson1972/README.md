
Convert the tranparent background to white

```ls -1 *.png |  cut -d . -f 1 | while read line; do convert -flatten $line.png $line.png ; done;

Convert images to a movie

```ffmpeg -i simpson_1972_%02d.png simpson_2917.mpg