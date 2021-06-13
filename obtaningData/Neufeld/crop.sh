ls -1 *.png | cut -d . -f 1 | while read line ; do convert $line.png -crop 878x756+0+0 ../cropped/$line.png; done
